import os



def train(config_path):
    from alignment.rlhf.trainner.exec_engine import ExecutionEngine
    engine = ExecutionEngine(
        config_path=config_path
    )
    engine.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Alignment argument parser")
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()   
    train(args.config_path)
