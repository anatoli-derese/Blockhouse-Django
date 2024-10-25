# Django Stock Prediction App

## Prerequisites
- Docker
- Docker Compose
- AWS Account (for RDS and deployment)

## Local Setup
1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a `.env` file based on `.env.example`:
    ```bash
    cp .env.example .env
    ```

3. Build and run the Docker containers:
    ```bash
    docker-compose up --build
    ```

4. Run migrations:
    ```bash
    docker-compose exec web python manage.py migrate
    ```

## Deployment
1. Ensure your AWS credentials are configured.
2. Push your changes to the `main` branch.
3. The application will be deployed automatically through GitHub Actions.