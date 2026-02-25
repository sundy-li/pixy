dev-cli:
	cargo run -p pixy-main --bin pixy

cli:
	cargo run -p pixy-main --bin pixy --release

dev-gateway:
	cargo run -p pixy-main --bin pixy -- gateway start

gateway:
	cargo run -p pixy-main --bin pixy --release -- gateway start

ut:
	cargo nextest run

release-check:
	cargo fmt --check
	cargo c
	cargo nextest run -p pixy-coding-agent -p pixy-main -p pixy-gateway
	cargo nextest run

install-local:
	./scripts/install.sh

bootstrap:
	./scripts/bootstrap.sh

onboard:
	./scripts/onboard.sh

package-manifests version:
	./scripts/generate-package-manifests.sh --version {{version}}
