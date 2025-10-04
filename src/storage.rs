#[derive(Clone, Debug)]
pub struct Storage { pub data: Vec<f32> }
impl Storage {
    pub fn as_mut(&mut self) -> &mut Self { self }
}
