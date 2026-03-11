use hex_core::{Cell, GameState, SIZE};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::Widget,
};

pub struct BoardView<'a> {
    pub state: &'a GameState,
    pub cursor: (u8, u8),
}

impl Widget for BoardView<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        for row in 0..SIZE {
            let y = area.y + row as u16;
            if y >= area.y + area.height {
                break;
            }
            let x_base = area.x + row as u16;
            for col in 0..SIZE {
                let x = x_base + col as u16 * 3;
                if x + 1 >= area.x + area.width {
                    break;
                }
                let cell = self.state.cell_at(row, col);
                let is_cursor = (row as u8, col as u8) == self.cursor;
                let (ch, style) = render_cell(cell, is_cursor);
                buf.set_string(x, y, " ", Style::default());
                buf.set_string(x + 1, y, ch, style);
            }
        }
    }
}

fn render_cell(cell: Cell, cursor: bool) -> (&'static str, Style) {
    if cursor {
        return ("*", Style::default().fg(Color::Yellow));
    }
    match cell {
        Cell::Empty => (".", Style::default().fg(Color::DarkGray)),
        Cell::Red => ("R", Style::default().fg(Color::Red)),
        Cell::Blue => ("B", Style::default().fg(Color::Blue)),
    }
}
