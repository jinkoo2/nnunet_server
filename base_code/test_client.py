import requests
import time

BASE_URL = "http://127.0.0.1:8000"  # FastAPI server URL


def submit_task(data):
    """
    Submits a task to the /submit-task endpoint.
    """
    response = requests.post(
        f"{BASE_URL}/submit-task/",
        json={"data": data},  # Task input as JSON payload
        headers={"Content-Type": "application/json"}
    )
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to submit task: {response.status_code}, {response.text}")
        return None


def check_task_status(task_id):
    """
    Checks the status of a submitted task using /task-status/{task_id}.
    """
    response = requests.get(f"{BASE_URL}/task-status/{task_id}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch task status: {response.status_code}, {response.text}")
        return None


if __name__ == "__main__":
    # Step 1: Submit a task
    task_data = "some_input_data"  # Replace with actual data to process
    print(f"Submitting task with data: {task_data}")
    result = submit_task(task_data)
    if result is None:
        print("Failed to submit the task.")
    else:
        task_id = result["task_id"]
        print(f"Task submitted successfully! Task ID: {task_id}")

        # Step 2: Poll for task status
        while True:
            print(f"Checking status of Task ID: {task_id}")
            status_result = check_task_status(task_id)
            if status_result is None:
                break

            task_status = status_result.get("status")
            print(f"Task Status: {task_status}")

            # Break out of loop if task is complete or failed
            if task_status in ["completed", "failed"]:
                print(f"Final Task Result: {status_result}")
                break

            # Wait before polling again
            time.sleep(5)
