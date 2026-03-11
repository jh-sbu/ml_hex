pub const SIZE: usize = 11;
pub const CELLS: usize = SIZE * SIZE; // 121

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cell {
    Empty = 0,
    Red = 1,
    Blue = 2,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub row: u8,
    pub col: u8,
}

impl Move {
    pub fn index(self) -> usize {
        self.row as usize * SIZE + self.col as usize
    }
}
