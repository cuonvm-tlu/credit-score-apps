from app.core.kafka_consumer import consume_messages


def main() -> None:
    """Main entry point for the Analytic Service."""
    print("Starting Analytic Service...")
    consume_messages()


if __name__ == "__main__":
    main()