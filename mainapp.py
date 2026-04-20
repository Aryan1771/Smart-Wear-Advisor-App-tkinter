from desktop_app.app import run
if __name__ == "__main__":
    try:
        run()
    except ModuleNotFoundError as error:
        print(error)
    except Exception as error:
        print(f"Application failed to start: {error}")
