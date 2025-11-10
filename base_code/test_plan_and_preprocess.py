import requests
import time

BASE_URL = "http://127.0.0.1:8000"  # FastAPI server URL

def submit_plan_and_preprocess_task(dataset_num, planner, verify_dataset_integrity):
    """
    Submits a task 
    """
    response = requests.post(
        f"{BASE_URL}/plan-and-preprocess/",
        json={
            "dataset_num": dataset_num,
            "planner": planner,
            "verify_dataset_integrity": verify_dataset_integrity,
            },  # Task input as JSON payload
        headers={"Content-Type": "application/json"}
    )
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to submit task: {response.status_code}, {response.text}")
        return None


def check_plan_and_preprocess_task_status(task_id):
    """
    Checks the status of a submitted task using /task-status/{task_id}.
    """
    response = requests.get(f"{BASE_URL}/celery/task-status/{task_id}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch task status: {response.status_code}, {response.text}")
        return None


if __name__ == "__main__":
    
    # Step 1: Submit a task
    dataset_num = 847
    planner ='ExperimentPlanner'
    verify_dataset_integrity = True

    result = submit_plan_and_preprocess_task(dataset_num, planner, verify_dataset_integrity)
    if result is None:
        print("Failed to submit the task.")
    else:
        print(f"Task submitted successfully!")

        