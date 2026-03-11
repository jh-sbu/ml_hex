use crate::cell::SIZE;

/// Returns up to 6 (row, col) neighbors of (r, c) on the 11×11 hex grid.
pub fn neighbors(r: usize, c: usize) -> impl Iterator<Item = (usize, usize)> {
    const DELTAS: [(i32, i32); 6] = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)];
    DELTAS.into_iter().filter_map(move |(dr, dc)| {
        let nr = r as i32 + dr;
        let nc = c as i32 + dc;
        if nr >= 0 && nr < SIZE as i32 && nc >= 0 && nc < SIZE as i32 {
            Some((nr as usize, nc as usize))
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neighbors_corner() {
        let n: Vec<_> = neighbors(0, 0).collect();
        assert_eq!(n.len(), 2);
    }

    #[test]
    fn test_neighbors_edge() {
        // (0,5): loses (-1,5) and (-1,6) [row -1 out of bounds], keeps 4 others
        let n: Vec<_> = neighbors(0, 5).collect();
        assert_eq!(n.len(), 4);
    }

    #[test]
    fn test_neighbors_center() {
        let n: Vec<_> = neighbors(5, 5).collect();
        assert_eq!(n.len(), 6);
    }
}
