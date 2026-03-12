use crate::board::neighbors;
use crate::cell::{Cell, Move, CELLS, SIZE};
use crate::dsu::{Dsu, BOTTOM, LEFT, RIGHT, TOP};
use crate::error::HexError;

#[derive(Clone, Debug)]
pub struct GameState {
    cells: [Cell; CELLS],
    dsu_red: Dsu,
    dsu_blue: Dsu,
    current_player: Cell, // Red or Blue (never Empty)
    move_count: u16,
    swap_available: bool,
    first_move: Option<Move>, // stored to support swap rule
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    pub fn new() -> Self {
        Self {
            cells: [Cell::Empty; CELLS],
            dsu_red: Dsu::new(),
            dsu_blue: Dsu::new(),
            current_player: Cell::Red,
            move_count: 0,
            swap_available: false,
            first_move: None,
        }
    }

    pub fn new_with_swap() -> Self {
        Self {
            swap_available: true,
            ..Self::new()
        }
    }

    pub fn current_player(&self) -> Cell {
        self.current_player
    }

    pub fn move_count(&self) -> u16 {
        self.move_count
    }

    pub fn cell_at(&self, row: usize, col: usize) -> Cell {
        self.cells[row * SIZE + col]
    }

    pub fn winner(&self) -> Option<Cell> {
        // Need mutable borrow for path compression; clone DSUs locally.
        let mut dr = self.dsu_red.clone();
        if dr.connected(TOP, BOTTOM) {
            return Some(Cell::Red);
        }
        let mut db = self.dsu_blue.clone();
        if db.connected(LEFT, RIGHT) {
            return Some(Cell::Blue);
        }
        None
    }

    pub fn is_terminal(&self) -> bool {
        self.winner().is_some()
    }

    /// Returns a winning move for the current player if one exists, or `None`.
    ///
    /// Runs in O(CELLS × 6) amortized time using in-place DSU path compression.
    /// Much cheaper than calling `apply_move` + `is_terminal` for each candidate.
    pub fn winning_move(&mut self) -> Option<Move> {
        match self.current_player {
            Cell::Red => {
                let root_top = self.dsu_red.find(TOP);
                let root_bot = self.dsu_red.find(BOTTOM);
                for i in 0..CELLS {
                    if self.cells[i] != Cell::Empty {
                        continue;
                    }
                    let r = i / SIZE;
                    let c = i % SIZE;
                    let mut at_top = r == 0;
                    let mut at_bot = r == SIZE - 1;
                    for (nr, nc) in neighbors(r, c) {
                        if self.cells[nr * SIZE + nc] == Cell::Red {
                            let root = self.dsu_red.find(nr * SIZE + nc);
                            if root == root_top {
                                at_top = true;
                            }
                            if root == root_bot {
                                at_bot = true;
                            }
                        }
                    }
                    if at_top && at_bot {
                        return Some(Move { row: r as u8, col: c as u8 });
                    }
                }
            }
            Cell::Blue => {
                let root_left = self.dsu_blue.find(LEFT);
                let root_right = self.dsu_blue.find(RIGHT);
                for i in 0..CELLS {
                    if self.cells[i] != Cell::Empty {
                        continue;
                    }
                    let r = i / SIZE;
                    let c = i % SIZE;
                    let mut at_left = c == 0;
                    let mut at_right = c == SIZE - 1;
                    for (nr, nc) in neighbors(r, c) {
                        if self.cells[nr * SIZE + nc] == Cell::Blue {
                            let root = self.dsu_blue.find(nr * SIZE + nc);
                            if root == root_left {
                                at_left = true;
                            }
                            if root == root_right {
                                at_right = true;
                            }
                        }
                    }
                    if at_left && at_right {
                        return Some(Move { row: r as u8, col: c as u8 });
                    }
                }
            }
            Cell::Empty => unreachable!(),
        }
        None
    }

    pub fn legal_moves(&self) -> impl Iterator<Item = Move> + '_ {
        (0..CELLS).filter_map(|i| {
            if self.cells[i] == Cell::Empty {
                Some(Move {
                    row: (i / SIZE) as u8,
                    col: (i % SIZE) as u8,
                })
            } else {
                None
            }
        })
    }

    pub fn apply_move(&self, mv: Move) -> Result<Self, HexError> {
        if self.is_terminal() {
            return Err(HexError::GameOver);
        }
        let r = mv.row as usize;
        let c = mv.col as usize;
        if r >= SIZE || c >= SIZE {
            return Err(HexError::OutOfBounds {
                row: mv.row,
                col: mv.col,
            });
        }
        if self.cells[mv.index()] != Cell::Empty {
            return Err(HexError::CellOccupied {
                row: mv.row,
                col: mv.col,
            });
        }

        let mut next = self.clone();
        next.cells[mv.index()] = self.current_player;

        let cell_idx = mv.index();
        match self.current_player {
            Cell::Red => {
                if r == 0 {
                    next.dsu_red.union(cell_idx, TOP);
                }
                if r == SIZE - 1 {
                    next.dsu_red.union(cell_idx, BOTTOM);
                }
                for (nr, nc) in neighbors(r, c) {
                    if next.cells[nr * SIZE + nc] == Cell::Red {
                        next.dsu_red.union(cell_idx, nr * SIZE + nc);
                    }
                }
            }
            Cell::Blue => {
                if c == 0 {
                    next.dsu_blue.union(cell_idx, LEFT);
                }
                if c == SIZE - 1 {
                    next.dsu_blue.union(cell_idx, RIGHT);
                }
                for (nr, nc) in neighbors(r, c) {
                    if next.cells[nr * SIZE + nc] == Cell::Blue {
                        next.dsu_blue.union(cell_idx, nr * SIZE + nc);
                    }
                }
            }
            Cell::Empty => unreachable!(),
        }

        // Swap remains available only right after Red's first move in swap-mode games.
        next.swap_available = self.swap_available && self.move_count == 0;
        next.current_player = if self.current_player == Cell::Red {
            Cell::Blue
        } else {
            Cell::Red
        };
        next.move_count += 1;
        if next.move_count == 1 {
            next.first_move = Some(mv);
        }

        Ok(next)
    }

    /// Swap rule: Blue mirrors Red's first move (flipping row/col).
    /// Only valid when swap_available == true (move_count == 1).
    pub fn apply_swap(&self) -> Result<Self, HexError> {
        if !self.swap_available || self.move_count != 1 {
            return Err(HexError::SwapNotAvailable);
        }
        let first = self.first_move.expect("first_move must be set at move_count==1");

        // Build fresh state and apply the mirrored move as Blue.
        // We start from a fresh board and place the swapped cell as Blue.
        let base = if self.swap_available {
            GameState::new_with_swap()
        } else {
            GameState::new()
        };

        // The swap: Blue takes the position (col, row) of the original Red move.
        let swap_mv = Move {
            row: first.col,
            col: first.row,
        };

        // Temporarily make the base state think it's Blue's turn (skip Red's move).
        // Simplest: create a state with Blue to move and apply the swap move.
        let mut blue_first = base;
        blue_first.current_player = Cell::Blue;
        blue_first.swap_available = false;
        blue_first.move_count = 1; // treat as if Red already moved (skipped)

        blue_first.apply_move(swap_mv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_board_no_winner() {
        assert_eq!(GameState::new().winner(), None);
    }

    #[test]
    fn test_red_wins() {
        // Red connects top (row 0) to bottom (row 10) via column 0.
        let mut state = GameState::new();
        for row in 0..SIZE as u8 {
            state = state
                .apply_move(Move { row, col: 0 })
                .expect("red move");
            if row < SIZE as u8 - 1 {
                // Blue plays somewhere safe (col 10)
                state = state
                    .apply_move(Move { row, col: 10 })
                    .expect("blue move");
            }
        }
        assert_eq!(state.winner(), Some(Cell::Red));
    }

    #[test]
    fn test_blue_wins() {
        // Blue connects left (col 0) to right (col 10) via row 0.
        // Red plays first so we interleave.
        let mut state = GameState::new();
        for col in 0..SIZE as u8 {
            // Red plays col 10 (bottom row won't form chain for now)
            state = state
                .apply_move(Move { row: 10, col })
                .expect("red move");
            // Blue plays row 0
            state = state
                .apply_move(Move { row: 0, col })
                .expect("blue move");
        }
        assert_eq!(state.winner(), Some(Cell::Blue));
    }

    #[test]
    fn test_occupied_cell_error() {
        let state = GameState::new();
        let mv = Move { row: 0, col: 0 };
        let next = state.apply_move(mv).unwrap();
        let err = next
            .apply_move(Move {
                row: 0,
                col: 0,
            })
            .unwrap_err();
        // Blue tries the same cell
        assert_eq!(err, HexError::CellOccupied { row: 0, col: 0 });
    }

    #[test]
    fn test_out_of_bounds_error() {
        let state = GameState::new();
        let err = state
            .apply_move(Move { row: 11, col: 0 })
            .unwrap_err();
        assert_eq!(err, HexError::OutOfBounds { row: 11, col: 0 });
    }

    #[test]
    fn test_legal_moves_count() {
        let state = GameState::new();
        assert_eq!(state.legal_moves().count(), 121);

        let s2 = state.apply_move(Move { row: 0, col: 0 }).unwrap();
        assert_eq!(s2.legal_moves().count(), 120);

        let s3 = s2.apply_move(Move { row: 1, col: 1 }).unwrap();
        assert_eq!(s3.legal_moves().count(), 119);
    }

    #[test]
    fn test_swap_rule() {
        // Red plays (3, 7). Blue swaps → Blue owns (7, 3).
        let state = GameState::new_with_swap();
        let after_red = state.apply_move(Move { row: 3, col: 7 }).unwrap();
        let after_swap = after_red.apply_swap().unwrap();

        // The swapped position should be Blue
        assert_eq!(after_swap.cell_at(7, 3), Cell::Blue);
        // Original Red position should be empty (fresh board was used)
        assert_eq!(after_swap.cell_at(3, 7), Cell::Empty);
        // It's now Red's turn (Blue just moved via swap)
        assert_eq!(after_swap.current_player(), Cell::Red);
    }

    #[test]
    fn test_immutability() {
        let state = GameState::new();
        let _next = state.apply_move(Move { row: 0, col: 0 }).unwrap();
        // Original is unchanged
        assert_eq!(state.cell_at(0, 0), Cell::Empty);
        assert_eq!(state.current_player(), Cell::Red);
    }

    #[test]
    fn test_clone_is_independent() {
        let state = GameState::new();
        let a = state.apply_move(Move { row: 0, col: 0 }).unwrap();
        let b = a.clone();
        let _a2 = a.apply_move(Move { row: 1, col: 0 }).unwrap(); // Blue in a (diverged)
        // b should not see Blue's move
        assert_eq!(b.cell_at(1, 0), Cell::Empty);
    }

    #[test]
    fn test_game_over_error() {
        let mut state = GameState::new();
        // Red wins along column 0
        for row in 0..SIZE as u8 {
            state = state.apply_move(Move { row, col: 0 }).expect("red");
            if row < SIZE as u8 - 1 {
                state = state.apply_move(Move { row, col: 10 }).expect("blue");
            }
        }
        assert!(state.is_terminal());
        let err = state.apply_move(Move { row: 5, col: 5 }).unwrap_err();
        assert_eq!(err, HexError::GameOver);
    }
}
