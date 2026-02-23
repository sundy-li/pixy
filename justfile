dev-cli:
	cargo run -p pi-coding-agent --bin pi

cli:
	cargo run -p pi-coding-agent --bin pi --release

ut:
	cargo t
