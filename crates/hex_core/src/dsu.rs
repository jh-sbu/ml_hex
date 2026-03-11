// Virtual node layout:
//   0..121  → board cells (row*11 + col)
//   121     → TOP wall (Red)
//   122     → BOTTOM wall (Red)
//   123     → LEFT wall (Blue)
//   124     → RIGHT wall (Blue)

pub const TOP: usize = 121;
pub const BOTTOM: usize = 122;
pub const LEFT: usize = 123;
pub const RIGHT: usize = 124;
const N: usize = 125;

#[derive(Clone, Debug)]
pub struct Dsu {
    parent: [u16; N],
    rank: [u8; N],
}

impl Dsu {
    pub fn new() -> Self {
        let mut parent = [0u16; N];
        for (i, slot) in parent.iter_mut().enumerate() {
            *slot = i as u16;
        }
        Self {
            parent,
            rank: [0u8; N],
        }
    }

    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] as usize != x {
            // Path compression (halving)
            let gp = self.parent[self.parent[x] as usize] as usize;
            self.parent[x] = gp as u16;
            x = self.parent[x] as usize;
        }
        x
    }

    pub fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            core::cmp::Ordering::Less => self.parent[ra] = rb as u16,
            core::cmp::Ordering::Greater => self.parent[rb] = ra as u16,
            core::cmp::Ordering::Equal => {
                self.parent[rb] = ra as u16;
                self.rank[ra] += 1;
            }
        }
    }

    pub fn connected(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }
}
