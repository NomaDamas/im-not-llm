//! The token embedding table, looked up on the host: the graphs start at
//! `inputs_embeds`, so a package ships the table as raw little-endian bf16
//! rows (`embeddings.bf16`) and this maps it instead of loading it.

use half::{bf16, f16};
use inl_core::{InlError, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub(crate) struct TokenTable {
    map: Mmap,
    hidden: usize,
    rows: usize,
    scale: f32,
}

impl TokenTable {
    /// `scale` multiplies each row before the fp16 cast.
    pub(crate) fn open(path: &Path, hidden: usize, scale: f32) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the package files are read-only in the cache and not
        // modified while the engine holds them.
        let map = unsafe { Mmap::map(&file)? };
        let row_bytes = hidden * 2;
        if map.len() % row_bytes != 0 {
            return Err(InlError::Inference(format!(
                "{}: {} bytes is not a whole number of {hidden}-wide bf16 rows",
                path.display(),
                map.len()
            )));
        }
        let rows = map.len() / row_bytes;
        Ok(Self {
            map,
            hidden,
            rows,
            scale,
        })
    }

    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    /// Write the rows of `ids` back to back into `out` (fp16).
    pub(crate) fn gather(&self, ids: &[u32], out: &mut [f16]) -> Result<()> {
        debug_assert!(out.len() >= ids.len() * self.hidden);
        for (&id, target) in ids.iter().zip(out.chunks_exact_mut(self.hidden)) {
            let id = id as usize;
            if id >= self.rows {
                return Err(InlError::Inference(format!(
                    "token id {id} is outside the {}-row embedding table",
                    self.rows
                )));
            }
            let bytes = &self.map[id * self.hidden * 2..(id + 1) * self.hidden * 2];
            for (value, pair) in target.iter_mut().zip(bytes.chunks_exact(2)) {
                let raw = bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]]));
                *value = f16::from_f32(raw.to_f32() * self.scale);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_scaled_then_cast_and_ids_are_bounds_checked() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("embeddings.bf16");
        let values = [1.0f32, -2.0, 0.5, 3.0, 0.25, -0.125];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        std::fs::write(&path, bytes).unwrap();
        let table = TokenTable::open(&path, 2, 4.0).unwrap();
        let mut out = vec![f16::ZERO; 4];
        table.gather(&[2, 0], &mut out).unwrap();
        let got: Vec<f32> = out.iter().map(|v| v.to_f32()).collect();
        assert_eq!(got, [1.0, -0.5, 4.0, -8.0]);
        assert!(table.gather(&[3], &mut out).is_err());
    }
}
