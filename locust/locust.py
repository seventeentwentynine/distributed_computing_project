from locust import HttpUser, task, between


class NewsReviewer(HttpUser):
    # Simulates a user waiting between 1 and 2 seconds between actions
    wait_time = between(1, 2)

    @task
    def classify_article(self):
        # The JSON payload your FastAPI expects
        payload = {
            "text": "Breaking: Scientists discover a new planet made of chocolate."
        }

        # Send a POST request to the /predict endpoint
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed! Status code: {response.status_code}")

    # Optional: You can add a second task to check the health endpoint
    @task(3)  # The '3' means this task runs 3 times more often than the one above
    def check_health(self):
        self.client.get("/health")