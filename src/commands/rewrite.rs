use std::path::PathBuf;

use crate::error::AppError;
use crate::output::Ctx;

pub fn run(_ctx: Ctx, _file: PathBuf, _in_place: bool) -> Result<(), AppError> {
    Err(AppError::Transient(
        "rewrite: inference pipeline is on the v0.2 milestone. See https://github.com/199-biotechnologies/writer/milestones".into()
    ))
}
