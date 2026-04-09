use crate::error::AppError;
use crate::output::Ctx;

pub fn run(_ctx: Ctx, _prompt: String) -> Result<(), AppError> {
    Err(AppError::Transient(
        "write: inference pipeline is on the v0.2 milestone. See https://github.com/199-biotechnologies/writer/milestones".into()
    ))
}
