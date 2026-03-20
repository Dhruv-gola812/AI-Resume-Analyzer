import pandas as pd
import random

random.seed(42)

categories = {
    "Data Science": {
        "skills": [
            "Python", "Pandas", "NumPy", "Scikit-learn", "Machine Learning",
            "Deep Learning", "TensorFlow", "PyTorch", "SQL", "Data Analysis",
            "Statistics", "Matplotlib", "Seaborn", "NLP", "Computer Vision"
        ],
        "roles": [
            "Data Scientist", "ML Engineer", "Data Analyst", "AI Engineer"
        ],
        "projects": [
            "built a customer churn prediction model",
            "created a sales forecasting dashboard",
            "developed an NLP-based sentiment analysis system",
            "trained an image classification model",
            "analyzed large datasets to generate business insights"
        ]
    },
    "Web Development": {
        "skills": [
            "HTML", "CSS", "JavaScript", "React", "Node.js", "Express.js",
            "MongoDB", "MySQL", "REST API", "Bootstrap", "Tailwind CSS",
            "Git", "Frontend Development", "Backend Development", "Next.js"
        ],
        "roles": [
            "Frontend Developer", "Backend Developer", "Full Stack Developer", "Web Developer"
        ],
        "projects": [
            "developed a responsive e-commerce website",
            "built a full stack blog platform",
            "created a portfolio website with animations",
            "designed a REST API for user authentication",
            "integrated payment gateway into a web application"
        ]
    },
    "Android Development": {
        "skills": [
            "Java", "Kotlin", "Android Studio", "XML", "Firebase", "SQLite",
            "Room Database", "MVVM", "Jetpack Compose", "Retrofit",
            "Material Design", "REST API", "Gradle"
        ],
        "roles": [
            "Android Developer", "Mobile App Developer"
        ],
        "projects": [
            "built a task manager Android app",
            "developed a chat application using Firebase",
            "created a food delivery mobile app",
            "integrated REST APIs into Android applications",
            "implemented offline storage with Room database"
        ]
    },
    "Java Developer": {
        "skills": [
            "Java", "Spring Boot", "Hibernate", "JPA", "MySQL", "PostgreSQL",
            "REST API", "Microservices", "Maven", "Gradle", "OOP",
            "JUnit", "Git", "Docker"
        ],
        "roles": [
            "Java Developer", "Backend Engineer", "Software Engineer"
        ],
        "projects": [
            "developed enterprise backend applications",
            "built RESTful services using Spring Boot",
            "designed microservices for scalable systems",
            "implemented authentication and authorization modules",
            "optimized database queries for performance"
        ]
    },
    "DevOps": {
        "skills": [
            "Docker", "Kubernetes", "Jenkins", "GitHub Actions", "AWS",
            "Linux", "Shell Scripting", "Terraform", "Ansible", "CI/CD",
            "Monitoring", "Prometheus", "Grafana", "Nginx"
        ],
        "roles": [
            "DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer"
        ],
        "projects": [
            "automated CI/CD pipelines for deployment",
            "containerized applications using Docker",
            "managed Kubernetes clusters in production",
            "set up monitoring dashboards with Grafana",
            "provisioned cloud infrastructure using Terraform"
        ]
    },
    "UI/UX Designer": {
        "skills": [
            "Figma", "Adobe XD", "Wireframing", "Prototyping", "User Research",
            "Design Systems", "Usability Testing", "Interaction Design",
            "Visual Design", "Responsive Design", "Photoshop", "Illustrator"
        ],
        "roles": [
            "UI Designer", "UX Designer", "Product Designer"
        ],
        "projects": [
            "designed mobile app user interfaces",
            "created wireframes and interactive prototypes",
            "conducted user research and usability testing",
            "built a scalable design system",
            "redesigned website flows to improve user experience"
        ]
    }
}

education_list = [
    "Bachelor of Technology in Computer Science",
    "Bachelor of Engineering in Information Technology",
    "Master of Computer Applications",
    "Bachelor of Science in Computer Science",
    "Diploma in Software Engineering"
]

experience_lines = [
    "Worked on real-world projects and collaborated with cross-functional teams.",
    "Improved application performance and user experience through optimization.",
    "Participated in agile development and version control workflows.",
    "Built scalable solutions and maintained clean code practices.",
    "Tested, debugged, and deployed production-ready applications."
]

def make_resume(category_name, info):
    role = random.choice(info["roles"])
    skills = random.sample(info["skills"], k=min(7, len(info["skills"])))
    projects = random.sample(info["projects"], k=2)
    education = random.choice(education_list)
    exp1 = random.choice(experience_lines)
    exp2 = random.choice(experience_lines)

    text = f"""
    Name: Candidate {random.randint(1000, 9999)}
    Role: {role}

    Professional Summary:
    Motivated and detail-oriented {role} with hands-on experience in software projects and problem solving.
    Skilled in {", ".join(skills[:4])} and passionate about building impactful solutions.

    Skills:
    {", ".join(skills)}

    Projects:
    - {projects[0]}
    - {projects[1]}

    Experience:
    - {exp1}
    - {exp2}

    Education:
    {education}
    """
    return " ".join(text.split())

rows = []

samples_per_category = 200   # 6 categories × 200 = 1200 rows

for category_name, info in categories.items():
    for _ in range(samples_per_category):
        rows.append({
            "resume_text": make_resume(category_name, info),
            "category": category_name
        })

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("resume_dataset.csv", index=False, encoding="utf-8")

print("Dataset saved as resume_dataset.csv")
print("Total rows:", len(df))
print(df.head())