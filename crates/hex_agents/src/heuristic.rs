use std::collections::BinaryHeap;
use std::cmp::Reverse;

use hex_core::{neighbors, Cell, GameState, Move, CELLS, SIZE};

use crate::Agent;

const INFTY: u32 = 1_000_000;

pub struct HeuristicAgent;

impl Agent for HeuristicAgent {
    fn name(&self) -> &str {
        "Heuristic"
    }

    fn select_move(&self, state: &GameState) -> Move {
        let player = state.current_player();
        let dist_from = dijkstra(state, player, &source_seeds(state, player));
        let dist_to = dijkstra(state, player, &target_seeds(state, player));

        let best = (0..CELLS)
            .filter(|&i| state.cell_at(i / SIZE, i % SIZE) == Cell::Empty)
            .min_by_key(|&i| dist_from[i].saturating_add(dist_to[i]))
            .expect("no legal moves: state is terminal");

        Move {
            row: (best / SIZE) as u8,
            col: (best % SIZE) as u8,
        }
    }
}

fn cell_weight(cell: Cell, player: Cell) -> u32 {
    if cell == player {
        0
    } else if cell == Cell::Empty {
        1
    } else {
        // opponent cell — impassable
        INFTY
    }
}

/// Returns indices of the "source" wall cells for `player`.
/// Red: top row (row 0); Blue: left column (col 0).
fn source_seeds(state: &GameState, player: Cell) -> Vec<usize> {
    match player {
        Cell::Red => (0..SIZE)
            .filter(|&c| state.cell_at(0, c) != Cell::Blue)
            .collect(),
        Cell::Blue => (0..SIZE)
            .filter(|&r| state.cell_at(r, 0) != Cell::Red)
            .map(|r| r * SIZE)
            .collect(),
        Cell::Empty => unreachable!(),
    }
}

/// Returns indices of the "target" wall cells for `player`.
/// Red: bottom row (row 10); Blue: right column (col 10).
fn target_seeds(state: &GameState, player: Cell) -> Vec<usize> {
    match player {
        Cell::Red => (0..SIZE)
            .filter(|&c| state.cell_at(SIZE - 1, c) != Cell::Blue)
            .map(|c| (SIZE - 1) * SIZE + c)
            .collect(),
        Cell::Blue => (0..SIZE)
            .filter(|&r| state.cell_at(r, SIZE - 1) != Cell::Red)
            .map(|r| r * SIZE + (SIZE - 1))
            .collect(),
        Cell::Empty => unreachable!(),
    }
}

fn dijkstra(state: &GameState, player: Cell, seeds: &[usize]) -> [u32; CELLS] {
    let mut dist = [INFTY; CELLS];
    let mut heap: BinaryHeap<(Reverse<u32>, usize)> = BinaryHeap::new();

    for &i in seeds {
        let r = i / SIZE;
        let c = i % SIZE;
        let w = cell_weight(state.cell_at(r, c), player);
        if w < dist[i] {
            dist[i] = w;
            heap.push((Reverse(w), i));
        }
    }

    while let Some((Reverse(d), u)) = heap.pop() {
        if d > dist[u] {
            continue; // stale entry
        }
        let r = u / SIZE;
        let c = u % SIZE;
        for (nr, nc) in neighbors(r, c) {
            let v = nr * SIZE + nc;
            let w = cell_weight(state.cell_at(nr, nc), player);
            if w == INFTY {
                continue;
            }
            let nd = d + w;
            if nd < dist[v] {
                dist[v] = nd;
                heap.push((Reverse(nd), v));
            }
        }
    }

    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex_core::GameState;

    #[test]
    fn test_heuristic_wins_obvious() {
        // Red has a chain at col 0, rows 0–9. Blue is at col 10.
        // Red needs (10,0) to win. It's Red's turn.
        let mut state = GameState::new();
        for row in 0..10u8 {
            // Red plays col 0
            state = state.apply_move(Move { row, col: 0 }).unwrap();
            // Blue plays col 10
            state = state.apply_move(Move { row, col: 10 }).unwrap();
        }
        // It's Red's turn; (10,0) completes the chain
        let agent = HeuristicAgent;
        let mv = agent.select_move(&state);
        assert_eq!(mv, Move { row: 10, col: 0 });
    }

    #[test]
    fn test_heuristic_corridor() {
        // Blue fills row 5, cols 0–9. (5,10) is the only gap in that row.
        // Red must pass through col 10 to connect top↔bottom.
        // The heuristic should play somewhere in col 10 (the only open corridor).
        let mut state = GameState::new();
        for col in 0..10u8 {
            // Red plays row 10 (neutral filler)
            state = state.apply_move(Move { row: 10, col }).unwrap();
            // Blue seals row 5
            state = state.apply_move(Move { row: 5, col }).unwrap();
        }
        // Red's turn. Row 5 cols 0-9 are Blue; the only path is through col 10.
        let agent = HeuristicAgent;
        let mv = agent.select_move(&state);
        assert_eq!(mv.col, 10, "Heuristic must play in the only open corridor (col 10)");
    }
}
