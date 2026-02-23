dev-cli:
	cargo run -p pixy-coding-agent --bin pixy

cli:
	cargo run -p pixy-coding-agent --bin pixy --release

ut:
	cargo t
