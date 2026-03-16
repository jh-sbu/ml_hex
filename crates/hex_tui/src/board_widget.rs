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
        const BOARD_Y_OFFSET: u16 = 1; // 1 label row above
        const BOARD_X_OFFSET: u16 = 2; // "B " left margin

        let red_style = Style::default().fg(Color::Red);
        let blue_style = Style::default().fg(Color::Blue);

        // Top Red label
        let label_x = area.x + BOARD_X_OFFSET + (SIZE as u16 * 3) / 2;
        buf.set_string(label_x, area.y, "Red", red_style);

        for row in 0..SIZE {
            let y = area.y + BOARD_Y_OFFSET + row as u16;
            if y >= area.y + area.height {
                break;
            }

            let x_base = area.x + BOARD_X_OFFSET + row as u16 * 2;

            // Left Blue label
            buf.set_string(area.x, y, "B", blue_style);

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

            // Right Blue label
            let right_x = x_base + SIZE as u16 * 3;
            if right_x < area.x + area.width {
                buf.set_string(right_x, y, "B", blue_style);
            }
        }

        // Bottom Red label
        let bottom_y = area.y + BOARD_Y_OFFSET + SIZE as u16;
        if bottom_y < area.y + area.height {
            buf.set_string(label_x, bottom_y, "Red", red_style);
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
