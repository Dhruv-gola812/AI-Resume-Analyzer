import io
import os
import re
import pickle
import traceback
from flask import Flask, render_template, request, jsonify

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

TESSERACT_CMD = os.getenv('TESSERACT_CMD', r"C:\Program Files\Tesseract-OCR\tesseract.exe")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "resume_category_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "resume_vectorizer.pkl")

app = Flask(__name__, template_folder='templates', static_folder='assets')

# Keep categories and skills in sync with make_resume_dataset.py
CATEGORY_SKILLS = {
    "Data Science": ["Python", "Pandas", "NumPy", "Scikit-learn", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "SQL", "Data Analysis", "Statistics", "Matplotlib", "Seaborn", "NLP", "Computer Vision"],
    "Web Development": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Express.js", "MongoDB", "MySQL", "REST API", "Bootstrap", "Tailwind CSS", "Git", "Frontend Development", "Backend Development", "Next.js"],
    "Android Development": ["Java", "Kotlin", "Android Studio", "XML", "Firebase", "SQLite", "Room Database", "MVVM", "Jetpack Compose", "Retrofit", "Material Design", "REST API", "Gradle"],
    "Java Developer": ["Java", "Spring Boot", "Hibernate", "JPA", "MySQL", "PostgreSQL", "REST API", "Microservices", "Maven", "Gradle", "OOP", "JUnit", "Git", "Docker"],
    "DevOps": ["Docker", "Kubernetes", "Jenkins", "GitHub Actions", "AWS", "Linux", "Shell Scripting", "Terraform", "Ansible", "CI/CD", "Monitoring", "Prometheus", "Grafana", "Nginx"],
    "UI/UX Designer": ["Figma", "Adobe XD", "Wireframing", "Prototyping", "User Research", "Design Systems", "Usability Testing", "Interaction Design", "Visual Design", "Responsive Design", "Photoshop", "Illustrator"]
}

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model and/or vectorizer artifact missing. Run train_model.py first.")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_skills(text, skill_list):
    text = text.lower()
    found = []
    for skill in skill_list:
        token = skill.lower()
        if re.search(rf"\b{re.escape(token)}\b", text):
            found.append(skill)
    return found


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    try:
        resume_text = request.form.get('text', '').strip()

        uploaded_file = request.files.get('file')
        if uploaded_file and uploaded_file.filename:
            filename = uploaded_file.filename.strip().lower()
            ext = os.path.splitext(filename)[1]

            if ext == '.txt':
                resume_text = uploaded_file.read().decode('utf-8', errors='ignore').strip()

            elif ext == '.pdf':
                if pdfplumber is None:
                    return jsonify({'error': 'pdfplumber is required for PDF text extraction. pip install pdfplumber'}), 500

                uploaded_file.stream.seek(0)
                try:
                    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                        resume_text = '\n'.join([page.extract_text() or '' for page in pdf.pages]).strip()
                    if not resume_text:
                        return jsonify({'error': 'No text extracted from PDF. Maybe it is scanned image PDF. Use JPG/PNG or convert to text pdf.'}), 400
                except Exception as e:
                    traceback.print_exc()
                    return jsonify({'error': 'Failed to read PDF: ' + str(e)}), 400

            elif ext in ['.jpg', '.jpeg', '.png']:
                if Image is None or pytesseract is None:
                    return jsonify({'error': 'Pillow and pytesseract are required for image OCR. pip install pillow pytesseract'}), 500

                if os.path.exists(TESSERACT_CMD):
                    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
                else:
                    return jsonify({'error': f'Tesseract binary not found at {TESSERACT_CMD}. Install Tesseract and set TESSERACT_CMD env var.'}), 500

                uploaded_file.stream.seek(0)
                try:
                    image = Image.open(io.BytesIO(uploaded_file.read()))
                    image = image.convert('RGB')
                    resume_text = pytesseract.image_to_string(image)
                    if not resume_text.strip():
                        return jsonify({'error': 'No text found in image. Try a clearer image or higher-quality scan.'}), 400
                except Exception as e:
                    traceback.print_exc()
                    return jsonify({'error': 'Failed to OCR image: ' + str(e)}), 400

            else:
                return jsonify({'error': 'Unsupported file type for server analysis. Supported: .txt, .pdf, .jpg, .jpeg, .png.'}), 400

        if not resume_text or len(resume_text) < 20:
            return jsonify({'error': 'Please supply at least 20 characters of resume text.'}), 400

        cleaned = clean_text(resume_text)
        X = vectorizer.transform([cleaned])

        predicted_category = model.predict(X)[0]

        score = 0
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(X)
                if hasattr(model, 'classes_'):
                    idx = list(model.classes_).index(predicted_category)
                    score = int(round(probs[0][idx] * 100))
            except Exception:
                score = 75

        if score <= 0:
            score = 75

        skills = CATEGORY_SKILLS.get(predicted_category, [])
        present_skills = detect_skills(cleaned, skills)
        missing_skills = [s for s in skills if s not in present_skills]

        summary = (
            f"Predicted category: {predicted_category}. "
            "Your resume has good alignment with this career path. "
            "Focus on adding missing key skills to improve ATS match and recruiter interest."
        )

        suggestions = []
        if missing_skills:
            suggestions.append(f"Mention {missing_skills[0]} in a project or skills section.")
            if len(missing_skills) > 1:
                suggestions.append(f"Include {missing_skills[1]} in your summary or achievements.")
            if len(missing_skills) > 2:
                suggestions.append(f"Add experience for {missing_skills[2]} to increase relevancy.")
        else:
            suggestions.append("Great job! Your resume already contains key category skills.")
            suggestions.append("Consider adding specific metrics and outcomes for each project.")

        return jsonify({
            'score': min(100, max(1, score)),
            'presentSkills': present_skills,
            'missingSkills': missing_skills,
            'summary': summary,
            'suggestions': suggestions
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Internal server error: ' + str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
