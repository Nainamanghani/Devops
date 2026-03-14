# CI/CD for Java Spring Energy App

**Group No:** D6-Group 02  
**Project No.:** DO-13  

---

## Project Description

The **CI/CD for Java Spring Energy App** project focuses on implementing a **Continuous Integration and Continuous Deployment (CI/CD) pipeline** for an energy monitoring application built using **Java Spring Boot**.

The main objective of the project is to automate the software development lifecycle, including **code integration, building, testing, containerization, and deployment**.

In this project, developers push their code to a **Git repository**, which triggers the **CI/CD pipeline** using **Jenkins or GitHub Actions**. The pipeline automatically builds the application using **Maven**, runs tests, and creates a **Docker container image**. This image is then deployed to a **Kubernetes cluster**, where the application runs in a scalable and managed environment.

By integrating tools like **Git, Docker, Jenkins/GitHub Actions, and Kubernetes**, the project demonstrates how **DevOps practices improve development efficiency, reduce manual errors, and ensure faster and reliable deployment**.

---

## Project Overview

This project demonstrates the implementation of a **CI/CD (Continuous Integration and Continuous Deployment) pipeline** for a **Java Spring Boot Energy Monitoring Application** using modern **DevOps tools**.

The goal of the project is to automate the **build, testing, containerization, and deployment** process of a Spring Boot application.

Whenever code changes are pushed to the **Git repository**, the application is automatically:

- Built
- Tested
- Containerized using Docker
- Deployed to a Kubernetes cluster

This helps in **reducing manual effort, improving deployment speed, and ensuring reliable software delivery**.

---

## Objectives

- **Automation:** Automates the process of building, testing, and deploying the Java Spring Energy application using a CI/CD pipeline.
- **Consistency:** Ensures the application is built and deployed in the same way every time, reducing manual errors.
- **Scalability:** Uses Kubernetes to manage and scale application containers efficiently based on demand.
- **Security:** Integrates secure version control and controlled deployment processes to protect the application and codebase.
- **Efficiency:** Reduces manual work and speeds up the development and deployment cycle.
- **Reliability:** Continuous integration and automated testing help maintain application stability and performance.

---

## Tech Stack

| Category | Tools / Technologies |
|--------|--------------------|
| Version Control | Git, GitHub |
| Programming Language | Java |
| Framework | Spring Boot |
| Build Tool | Maven |
| Containerization | Docker |
| CI/CD | Jenkins / GitHub Actions |
| Container Orchestration | Kubernetes (K8s) |
| Deployment Platform | Kubernetes Cluster |

---

## Key Features

### Automated Infrastructure
The CI/CD pipeline automatically builds, tests, and deploys the application using tools like **Jenkins/GitHub Actions, Docker, and Kubernetes**, reducing manual setup.

### Secure Authentication
Secure access to the **Git repository, CI/CD pipeline, and container registry** ensures only authorized users can modify or deploy the application.

### State Management
Kubernetes maintains the desired state of application containers, ensuring the system automatically restores or restarts containers if failures occur.

### Continuous Deployment
Whenever new code is pushed to the Git repository, the pipeline automatically builds, tests, and deploys the updated application.

### Microservice Hosting
The application runs inside **Docker containers** and is managed by **Kubernetes**, enabling scalable and efficient hosting of services.

---

## Group Members

1. **Mahak Shrimal** – EN22CS301566  
2. **Mahak Taose** – EN22CS301567  
3. **Naina Manghani** – EN22CS301626  
4. **Mohit Patel** – EN22CS301610  
5. **Paavan Patni** – EN22CS301671  

---

## Conclusion

This project demonstrates the implementation of a **CI/CD pipeline for a Java Spring Energy application using DevOps tools**.

By integrating tools like **Git, Docker, Jenkins/GitHub Actions, and Kubernetes**, the project automates the processes of **building, testing, containerizing, and deploying the application**.

The use of CI/CD practices improves **development efficiency, deployment speed, and application reliability while reducing manual errors**. Containerization with Docker and orchestration with Kubernetes ensure **scalable and consistent deployment of the application**.

Overall, this project highlights the **importance of DevOps practices in modern software development and delivery**.