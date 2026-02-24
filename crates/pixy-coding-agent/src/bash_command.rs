use std::borrow::Cow;

pub(crate) fn normalize_nested_bash_lc(command: &str) -> Cow<'_, str> {
    if let Some(inner_command) = unwrap_nested_bash_lc(command.trim()) {
        return Cow::Owned(inner_command);
    }
    Cow::Borrowed(command)
}

fn unwrap_nested_bash_lc(command: &str) -> Option<String> {
    let parts = shlex::split(command)?;
    if parts.len() != 3 {
        return None;
    }

    if !is_bash_program(&parts[0]) || parts[1] != "-lc" {
        return None;
    }

    Some(parts[2].clone())
}

fn is_bash_program(program: &str) -> bool {
    program == "bash" || program.ends_with("/bash")
}

#[cfg(test)]
mod tests {
    use super::normalize_nested_bash_lc;

    #[test]
    fn unwraps_nested_bash_lc_single_quoted_commands() {
        let normalized = normalize_nested_bash_lc("bash -lc 'printf \"hello\"'");
        assert_eq!(normalized.as_ref(), "printf \"hello\"");
    }

    #[test]
    fn unwraps_nested_bash_lc_double_quoted_commands() {
        let normalized = normalize_nested_bash_lc("/bin/bash -lc \"echo hi\"");
        assert_eq!(normalized.as_ref(), "echo hi");
    }

    #[test]
    fn keeps_non_wrapped_commands_as_is() {
        let normalized = normalize_nested_bash_lc("cat /tmp/example.txt");
        assert_eq!(normalized.as_ref(), "cat /tmp/example.txt");
    }
}
