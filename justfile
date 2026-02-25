dev-cli:
	cargo run -p pixy-coding-agent --bin pixy

cli:
	cargo run -p pixy-coding-agent --bin pixy --release

dev-gateway:
	cargo run -p pixy-gateway --bin pixy-gateway -- start

gateway:
	cargo run -p pixy-gateway --bin pixy-gateway --release -- start

ut:
	cargo nextest run
