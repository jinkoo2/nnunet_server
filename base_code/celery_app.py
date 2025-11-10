# from celery import Celery

# # Configure Celery to use Redis as the message broker
# celery_app = Celery(
#     "tasks",
#     broker="redis://localhost:6379/0",  # Redis as the message broker
#     backend="redis://localhost:6379/0" # Redis as the results backend
# )

# # Define Celery tasks
# @celery_app.task
# def long_task(data):
#     import time
#     time.sleep(30)  # Simulate a lengthy process (1 min)
#     return {"status": "completed", "result": f"Processed {data}"}

# @celery_app.task
# def nnunet_plan_and_preprocess(dataset_num, planner, verify_dataset_integrity):
#     print('=== celery_app.task ===')
#     print(f'nnunet_plan_and_preprocess({dataset_num}, {planner}, {verify_dataset_integrity})')
#     from fastapi.server.nnunet_plan_and_preprocess import plan_and_preprocess_sh
#     plan_and_preprocess_sh(dataset_num,planner, verify_dataset_integrity)
#     return {"status": "completed", "result": f"Processed {dataset_num}"}

