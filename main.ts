// Sokoban push-optimal solver, written from scratch for this competition.
//
// Idea: a level is one 1-wide hallway with 1-4 rooms hanging off it, each
// room opening through a single door. A box pushed out of a room can never be
// pushed back (the player would have to stand on the hallway's far side,
// which is wall), so every room is an independent problem and the level's
// push optimum is the sum of the rooms' optima. This program therefore
// splits a level into rooms, solves each room push-optimally from its door,
// and stitches the room solutions together with shortest walks along the
// hallway. Walks cost no pushes, so the total push count equals P* when
// every room is solved push-optimally.
//
// Each room is solved with A* over pushes. A state is the sorted box cells
// plus the player's reachable region (canonicalised to its lowest cell);
// a step costs 1 per push and 0 per walk. The heuristic is the minimum
// assignment (Hungarian, brute-forced for <= 4 boxes) of boxes to goals over
// walking distances, which is admissible and consistent for push cost.
// Pruning: static dead cells from a reverse pull-reachability analysis
// (a box on a cell from which no goal is pull-reachable can never be solved),
// dynamic frozen-box detection (iterative fixed point) and 2x2 wall/box
// blocks. Budgets are counted in expanded states, never in wall-clock time,
// so two runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in fixed order: up, down, left, right. Opposite of d is d ^ 1.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];
const PUSH_LETTER = ['U', 'D', 'L', 'R'];

// Work budgets (expanded states). Deterministic: no clock is ever read.
const ASTAR_BUDGET = 1_500_000;
const WEIGHTED_BUDGET = 2_000_000;
const GREEDY_BUDGET = 2_000_000;

interface Parsed {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Parsed {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x] as string;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Shortest walk (BFS) from `from` to `to` on the full board, treating walls
// and the given boxes as obstacles. Returns the move letters, or null.
function walkPath(
  w: number,
  h: number,
  wall: Uint8Array,
  boxes: number[],
  from: number,
  to: number,
): string | null {
  if (from === to) return '';
  const blocked = new Uint8Array(w * h);
  for (const b of boxes) blocked[b] = 1;
  const prev = new Int32Array(w * h).fill(-1);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi] as number;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + (DX[d] as number);
      const ny = cy + (DY[d] as number);
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const q = ny * w + nx;
      if (wall[q] || blocked[q] || prev[q] !== -1) continue;
      prev[q] = c;
      if (q === to) {
        // Reconstruct.
        let moves = '';
        let cur = q;
        const steps: number[] = [];
        while (cur !== from) {
          steps.push(cur);
          cur = prev[cur] as number;
        }
        steps.reverse();
        let at = from;
        for (const s of steps) {
          const dx = (s % w) - (at % w);
          const dy = ((s / w) | 0) - ((at / w) | 0);
          moves += dx === 1 ? 'r' : dx === -1 ? 'l' : dy === 1 ? 'd' : 'u';
          at = s;
        }
        return moves;
      }
      queue.push(q);
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// One room, solved on its own. Cells are reindexed 0..n-1; `cells[i]` is the
// full-board index of local cell i. The last local cell is the hallway square
// outside the door (the entry): the player starts there, boxes may never be
// pushed onto it, and the player may walk on it.
// ---------------------------------------------------------------------------

interface Room {
  n: number;
  w: number;
  cells: number[];
  // nb4[i][d] = local index of the neighbour of i in direction d, or -1.
  nb4: number[][];
  goals: number[];
  isGoal: Uint8Array;
  boxes: number[];
  player: number;
  // dist[g][c]: walking distance from goal g to cell c, ignoring boxes.
  distGoal: Int32Array[];
  // live[c]: 1 when a box on c can still reach a goal (pull-reachability).
  live: Uint8Array;
}

function buildRoom(
  w: number,
  roomCells: number[],
  entry: number,
  goals: number[],
  boxes: number[],
): Room {
  const cells = [...roomCells, entry];
  const n = cells.length;
  const idx = new Map<number, number>();
  cells.forEach((c, i) => idx.set(c, i));
  const nb4: number[][] = [];
  for (let i = 0; i < n; i++) {
    const c = cells[i] as number;
    const cx = c % w;
    const cy = (c / w) | 0;
    const row: number[] = [];
    for (let d = 0; d < 4; d++) {
      const nx = cx + (DX[d] as number);
      const ny = cy + (DY[d] as number);
      if (nx < 0 || ny < 0) {
        row.push(-1);
        continue;
      }
      const q = ny * w + nx;
      const j = idx.get(q);
      row.push(j === undefined ? -1 : j);
    }
    nb4.push(row);
  }
  const isGoal = new Uint8Array(n);
  const goalLocal: number[] = [];
  for (const g of goals) {
    const j = idx.get(g);
    if (j !== undefined) {
      isGoal[j] = 1;
      goalLocal.push(j);
    }
  }
  const boxLocal = boxes
    .map((b) => idx.get(b) as number)
    .filter((j) => j !== undefined)
    .sort((a, b) => a - b);
  const player = idx.get(entry) as number;

  // Walking distances from each goal, ignoring boxes.
  const distGoal: Int32Array[] = [];
  for (const g of goalLocal) {
    const dist = new Int32Array(n).fill(-1);
    const queue: number[] = [g];
    dist[g] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const c = queue[qi] as number;
      for (let d = 0; d < 4; d++) {
        const q = (nb4[c] as number[])[d] as number;
        if (q >= 0 && dist[q] < 0) {
          dist[q] = (dist[c] as number) + 1;
          queue.push(q);
        }
      }
    }
    distGoal.push(dist);
  }

  // Static dead cells: reverse pull-reachability from the goals, one box,
  // no other boxes. State (box, player); walk moves the player around the
  // box, a pull moves the box into the player's old cell. Any box cell the
  // reverse search never visits can never reach a goal, even with the other
  // boxes removed, so it is dead for sure.
  const live = new Uint8Array(n);
  {
    const visited = new Uint8Array(n * n);
    const queue: number[] = [];
    for (const g of goalLocal) {
      for (let p = 0; p < n; p++) {
        if (p === g) continue;
        const s = g * n + p;
        if (!visited[s]) {
          visited[s] = 1;
          queue.push(s);
        }
      }
    }
    for (let qi = 0; qi < queue.length; qi++) {
      const s = queue[qi] as number;
      const b = (s / n) | 0;
      const p = s % n;
      // Walk: player steps to an adjacent cell that is not the box.
      for (let d = 0; d < 4; d++) {
        const q = (nb4[p] as number[])[d] as number;
        if (q < 0 || q === b) continue;
        const s2 = b * n + q;
        if (!visited[s2]) {
          visited[s2] = 1;
          queue.push(s2);
        }
      }
      // Pull (reverse push): the box is adjacent to the player; the
      // predecessor has the box where the player stands and the player on
      // the far side of it.
      for (let d = 0; d < 4; d++) {
        if ((nb4[p] as number[])[d] !== b) continue;
        const pp = (nb4[p] as number[])[d ^ 1] as number;
        if (pp < 0) continue;
        const s2 = p * n + pp;
        if (!visited[s2]) {
          visited[s2] = 1;
          queue.push(s2);
        }
      }
    }
    for (let s = 0; s < n * n; s++) {
      if (visited[s]) live[(s / n) | 0] = 1;
    }
  }

  return { n, w, cells, nb4, goals: goalLocal, isGoal, boxes: boxLocal, player, distGoal, live };
}

// Minimum-assignment heuristic: the cheapest bijection boxes -> goals over
// walking distances. Admissible (each push moves one box one step, so the
// assignment cost drops by at most 1 per push) and consistent.
function assignmentHeuristic(room: Room, boxes: number[]): number {
  const k = boxes.length;
  const goals = room.goals;
  if (k !== goals.length) return Infinity;
  // Brute-force all bijections (k <= 4, at most 24). perm holds indices
  // into room.goals / room.distGoal, not cell values.
  const perm = goals.map((_, i) => i);
  let best = Infinity;
  const rec = (i: number, acc: number): void => {
    if (acc >= best) return;
    if (i === k) {
      best = acc;
      return;
    }
    for (let j = i; j < k; j++) {
      const tmp = perm[i] as number;
      perm[i] = perm[j] as number;
      perm[j] = tmp;
      const d = (room.distGoal[perm[i] as number] as Int32Array)[boxes[i] as number] as number;
      if (d >= 0) rec(i + 1, acc + d);
      const tmp2 = perm[i] as number;
      perm[i] = perm[j] as number;
      perm[j] = tmp2;
    }
  };
  rec(0, 0);
  return best;
}

// Deadlock test on a box configuration (sorted local cells):
// - any box on a statically dead cell, or
// - any 2x2 block of walls/boxes holding a box off its goal, or
// - any box off its goal frozen (no push direction open, where a direction
//   is open only if its destination and its stand cell are free of walls
//   and of frozen boxes; iterated to a fixed point).
function isDead(room: Room, boxes: number[]): boolean {
  const n = room.n;
  const boxSet = new Uint8Array(n);
  for (const b of boxes) boxSet[b] = 1;
  for (const b of boxes) {
    if (!room.live[b]) return true;
  }
  // 2x2 blocks: for each box off its goal, each 2x2 square holding it whose
  // other three cells are walls or boxes freezes it (every push needs a
  // stand cell or a destination inside the block, and all four are blocked;
  // the same holds for every other box in the block, so it never dissolves).
  // The two sides of the square must be perpendicular directions.
  for (const b of boxes) {
    if (room.isGoal[b]) continue;
    for (const d1 of [0, 1]) {
      for (const d2 of [2, 3]) {
        const c1 = (room.nb4[b] as number[])[d1] as number;
        const c2 = (room.nb4[b] as number[])[d2] as number;
        // The diagonal cell: step d2 from c1, or d1 from c2; when both
        // neighbours are walls the box is cornered whatever the diagonal is.
        const diag =
          c1 >= 0
            ? ((room.nb4[c1] as number[])[d2] as number)
            : c2 >= 0
              ? ((room.nb4[c2] as number[])[d1] as number)
              : -1;
        const blocking = (c: number): boolean => c < 0 || boxSet[c] === 1;
        if (blocking(c1) && blocking(c2) && blocking(diag)) return true;
      }
    }
  }
  // Frozen boxes, fixed point. A box is frozen when every push direction
  // is blocked by a wall or by a frozen box (destination or stand cell).
  const frozen = new Uint8Array(n);
  let changed = true;
  while (changed) {
    changed = false;
    for (const b of boxes) {
      if (frozen[b]) continue;
      let allBlocked = true;
      for (let d = 0; d < 4; d++) {
        const dest = (room.nb4[b] as number[])[d] as number;
        const stand = (room.nb4[b] as number[])[d ^ 1] as number;
        const destBlocked = dest < 0 || frozen[dest] === 1;
        const standBlocked = stand < 0 || frozen[stand] === 1;
        if (!destBlocked && !standBlocked) {
          allBlocked = false;
          break;
        }
      }
      if (allBlocked) {
        frozen[b] = 1;
        changed = true;
      }
    }
  }
  for (const b of boxes) {
    if (!room.isGoal[b] && frozen[b]) return true;
  }
  return false;
}

// Pack sorted box cells (each < 64) into one number, 6 bits per box.
function packBoxes(boxes: number[]): number {
  let v = 0;
  for (let i = boxes.length - 1; i >= 0; i--) v = (v << 6) | (boxes[i] as number);
  return v;
}

function unpackBoxes(packed: number, k: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < k; i++) out.push((packed >> (6 * i)) & 63);
  return out;
}

// Push-space search (A* with weight; weight 1 is optimal, a huge weight is
// greedy). Returns the move string (walks in lower case, pushes in upper
// case) and the push count, or null when the budget runs out.
function searchRoom(
  room: Room,
  weight: number,
  budget: number,
): { moves: string; pushes: number; expanded: number } | null {
  const k = room.boxes.length;
  const n = room.n;
  const startPacked = packBoxes(room.boxes);

  // Region of a state: BFS from the player cell avoiding boxes. Returns the
  // region's lowest cell plus membership and predecessor arrays (for walks).
  const regionOf = (
    boxes: number[],
    player: number,
  ): { rep: number; member: Uint8Array; prev: Int32Array } => {
    const boxSet = new Uint8Array(n);
    for (const b of boxes) boxSet[b] = 1;
    const member = new Uint8Array(n);
    const prev = new Int32Array(n).fill(-1);
    const queue: number[] = [player];
    member[player] = 1;
    prev[player] = player;
    let rep = player;
    for (let qi = 0; qi < queue.length; qi++) {
      const c = queue[qi] as number;
      if (c < rep) rep = c;
      for (let d = 0; d < 4; d++) {
        const q = (room.nb4[c] as number[])[d] as number;
        if (q < 0 || boxSet[q] || member[q]) continue;
        member[q] = 1;
        prev[q] = c;
        queue.push(q);
      }
    }
    return { rep, member, prev };
  };

  const hCache = new Map<number, number>();
  const heuristic = (packed: number): number => {
    const hit = hCache.get(packed);
    if (hit !== undefined) return hit;
    const h = assignmentHeuristic(room, unpackBoxes(packed, k));
    hCache.set(packed, h);
    return h;
  };
  const repCache = new Map<number, number>();
  const repOf = (packed: number, player: number): number => {
    const key = (packed << 6) | player;
    const hit = repCache.get(key);
    if (hit !== undefined) return hit;
    const rep = regionOf(unpackBoxes(packed, k), player).rep;
    repCache.set(key, rep);
    return rep;
  };

  const startH = heuristic(startPacked);
  if (startH === Infinity || isDead(room, room.boxes)) return null;

  // State storage (parallel arrays, indexed by state id).
  const stPacked: number[] = [startPacked];
  const stPlayer: number[] = [room.player];
  const stRep: number[] = [repOf(startPacked, room.player)];
  const stG: number[] = [0];
  const stH: number[] = [startH];
  const stParent: number[] = [-1];
  const stMoves: string[] = [''];
  const stSeq: number[] = [0];
  let seq = 1;

  const keyOf = (id: number): number => ((stPacked[id] as number) << 6) | (stRep[id] as number);
  const best = new Map<number, number>([[keyOf(0), 0]]);

  // Binary heap of state ids, ordered by (g + weight*h, -g, seq).
  const heap: number[] = [0];
  const fOf = (id: number): number => (stG[id] as number) + weight * (stH[id] as number);
  const less = (a: number, b: number): boolean => {
    const fa = fOf(a);
    const fb = fOf(b);
    if (fa !== fb) return fa < fb;
    if (stG[a] !== stG[b]) return (stG[a] as number) > (stG[b] as number);
    return (stSeq[a] as number) < (stSeq[b] as number);
  };
  const heapPush = (id: number): void => {
    let i = heap.length;
    heap.push(id);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i] as number, heap[p] as number)) break;
      const tmp = heap[i] as number;
      heap[i] = heap[p] as number;
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0] as number;
    const last = heap.pop() as number;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1] as number, heap[c] as number)) c++;
        if (!less(heap[c] as number, heap[i] as number)) break;
        const tmp = heap[i] as number;
        heap[i] = heap[c] as number;
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };

  const solved = (packed: number): boolean => {
    const boxes = unpackBoxes(packed, k);
    for (const b of boxes) if (!room.isGoal[b]) return false;
    return true;
  };

  let expanded = 0;
  while (heap.length > 0) {
    const s = heapPop();
    if (best.get(keyOf(s)) !== s) continue; // stale heap entry
    if (solved(stPacked[s] as number)) {
      let moves = '';
      let pushes = 0;
      for (let t = s; t > 0; t = stParent[t] as number) {
        moves = (stMoves[t] as string) + moves;
        pushes++;
      }
      return { moves, pushes, expanded };
    }
    expanded++;
    if (expanded > budget) return null;
    const boxes = unpackBoxes(stPacked[s] as number, k);
    const player = stPlayer[s] as number;
    const g = stG[s] as number;
    const region = regionOf(boxes, player);
    const boxSet = new Uint8Array(n);
    for (const b of boxes) boxSet[b] = 1;

    for (let bi = 0; bi < k; bi++) {
      const b = boxes[bi] as number;
      for (let d = 0; d < 4; d++) {
        const stand = (room.nb4[b] as number[])[d ^ 1] as number;
        const dest = (room.nb4[b] as number[])[d] as number;
        if (stand < 0 || dest < 0) continue;
        if (!region.member[stand]) continue;
        if (boxSet[dest]) continue;
        const nb = boxes.slice();
        nb[bi] = dest;
        nb.sort((a, b2) => a - b2);
        const packed2 = packBoxes(nb);
        const h2 = heuristic(packed2);
        if (h2 === Infinity) continue;
        if (isDead(room, nb)) continue;
        const rep2 = repOf(packed2, b);
        const key2 = (packed2 << 6) | rep2;
        const prevBest = best.get(key2);
        if (prevBest !== undefined && (stG[prevBest] as number) <= g + 1) continue;
        // Walk from the player to the stand cell, then the push.
        let walk = '';
        {
          const steps: number[] = [];
          let cur = stand;
          while (cur !== player) {
            steps.push(cur);
            cur = region.prev[cur] as number;
          }
          steps.reverse();
          let at = player;
          for (const step of steps) {
            const fc = room.cells[at] as number;
            const tc = room.cells[step] as number;
            const dx = (tc % room.w) - (fc % room.w);
            const dy = ((tc / room.w) | 0) - ((fc / room.w) | 0);
            walk += dx === 1 ? 'r' : dx === -1 ? 'l' : dy === 1 ? 'd' : 'u';
            at = step;
          }
        }
        const id = stPacked.length;
        stPacked.push(packed2);
        stPlayer.push(b);
        stRep.push(rep2);
        stG.push(g + 1);
        stH.push(h2);
        stParent.push(s);
        stMoves.push(walk + (PUSH_LETTER[d] as string));
        stSeq.push(seq++);
        best.set(key2, id);
        heapPush(id);
      }
    }
  }
  return null;
}

// Solve one room push-optimally (A* first, then weighted, then greedy
// fallbacks so a hard room still yields a solution, just not an optimal one).
function solveRoom(room: Room): { moves: string; pushes: number } | null {
  const a = searchRoom(room, 1, ASTAR_BUDGET);
  if (a !== null) return a;
  const b = searchRoom(room, 5, WEIGHTED_BUDGET);
  if (b !== null) return b;
  const c = searchRoom(room, 1_000_000, GREEDY_BUDGET);
  if (c !== null) return c;
  return null;
}

// ---------------------------------------------------------------------------
// Whole level: split into rooms off the hallway, solve each, stitch walks.
// ---------------------------------------------------------------------------

function solveLevel(text: string): string {
  const { w, h, wall, goals, boxes, player } = parseLevel(text);
  if (player < 0) return '';
  if (boxes.every((b) => goals.includes(b))) return '';
  const hallRow = (player / w) | 0;

  // Doors: the open cells in the wall rows directly above / below the
  // hallway. Each door belongs to one room.
  const doors: number[] = [];
  for (const r of [hallRow - 1, hallRow + 1]) {
    if (r < 0 || r >= h) continue;
    for (let x = 0; x < w; x++) {
      const c = r * w + x;
      if (!wall[c]) doors.push(c);
    }
  }
  doors.sort((a, b) => a - b);

  let curBoxes = boxes.slice();
  let curPlayer = player;
  let moves = '';

  for (const door of doors) {
    const doorX = door % w;
    const entry = hallRow * w + doorX;
    if (curBoxes.includes(entry)) return ''; // a box lost in the hallway
    // The room: flood fill from the door without stepping onto the hallway.
    const seen = new Uint8Array(w * h);
    const roomCells: number[] = [];
    const stack: number[] = [door];
    seen[door] = 1;
    while (stack.length > 0) {
      const c = stack.pop() as number;
      roomCells.push(c);
      const cx = c % w;
      const cy = (c / w) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = cx + (DX[d] as number);
        const ny = cy + (DY[d] as number);
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        if (ny === hallRow) continue;
        const q = ny * w + nx;
        if (wall[q] || seen[q]) continue;
        seen[q] = 1;
        stack.push(q);
      }
    }
    const rBoxes = curBoxes.filter((b) => seen[b]);
    const rGoals = goals.filter((g) => seen[g]);
    if (rBoxes.length === 0) continue;
    if (rBoxes.length !== rGoals.length) return '';

    // Walk to this room's door, then solve the room from the entry square.
    const toDoor = walkPath(w, h, wall, curBoxes, curPlayer, entry);
    if (toDoor === null) return '';
    moves += toDoor;
    curPlayer = entry;

    const room = buildRoom(w, roomCells, entry, rGoals, rBoxes);
    const solved = solveRoom(room);
    if (solved === null) return '';
    // Replay the room's moves on the full board to keep the box positions
    // and the player exact (and to prove the moves are legal there).
    for (const ch of solved.moves) {
      const lower = ch.toLowerCase();
      const d = lower === 'u' ? 0 : lower === 'd' ? 1 : lower === 'l' ? 2 : 3;
      const dd = (DY[d] as number) * w + (DX[d] as number);
      const next = curPlayer + dd;
      if (wall[next]) return '';
      const at = curBoxes.indexOf(next);
      if (at >= 0) {
        const beyond = next + dd;
        if (wall[beyond] || curBoxes.includes(beyond)) return '';
        curBoxes[at] = beyond;
        curBoxes.sort((a, b) => a - b);
      }
      curPlayer = next;
    }
    moves += solved.moves;

    // Walk back out to the entry square for the next room.
    const out = walkPath(w, h, wall, curBoxes, curPlayer, entry);
    if (out === null) return '';
    moves += out;
    curPlayer = entry;
  }

  // Final check: every box on a goal.
  if (!curBoxes.every((b) => goals.includes(b))) return '';
  return moves.toLowerCase();
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
