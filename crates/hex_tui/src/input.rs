use crossterm::event::{KeyCode, KeyEvent};

pub enum AppAction {
    MoveCursor(i8, i8),
    PlaceMove,
    Quit,
    None,
}

pub fn key_to_action(key: KeyEvent) -> AppAction {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => AppAction::MoveCursor(-1, 0),
        KeyCode::Down | KeyCode::Char('j') => AppAction::MoveCursor(1, 0),
        KeyCode::Left | KeyCode::Char('h') => AppAction::MoveCursor(0, -1),
        KeyCode::Right | KeyCode::Char('l') => AppAction::MoveCursor(0, 1),
        KeyCode::Enter | KeyCode::Char(' ') => AppAction::PlaceMove,
        KeyCode::Char('q') | KeyCode::Esc => AppAction::Quit,
        _ => AppAction::None,
    }
}
