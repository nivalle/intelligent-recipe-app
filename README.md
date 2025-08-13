# 🍳 Intelligent Recipe App

A FastAPI-based web application for intelligent recipe management.

## Features

- RESTful API with FastAPI
- Static file serving
- Modern web interface
- Interactive API documentation
- Real-time API testing

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd intelligent-recipe-app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## 📖 Usage

Once the application is running, you can access:

- **Main application**: http://localhost:8000/
- **API endpoint**: http://localhost:8000/api/hello
- **Interactive API documentation**: http://localhost:8000/docs
- **Alternative API docs**: http://localhost:8000/redoc

## 🔌 API Endpoints

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| `GET` | `/api/hello` | Returns a greeting message | `{"message": "Hello, Recipe App!"}` |
| `GET` | `/` | Serves the main HTML page | HTML content |

### Example API Response

```json
{
  "message": "Hello, Recipe App!"
}
```

## 🏗️ Project Structure

```
intelligent-recipe-app/
├── main.py              # FastAPI application entry point
├── index.html           # Main HTML page with interactive features
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
├── README.md           # Project documentation
└── venv/               # Virtual environment (not in git)
```

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python web framework)
- **Server**: Uvicorn (ASGI server)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **API Documentation**: Auto-generated Swagger UI
- **Development**: Python 3.12+

## 🔧 Development

### Running in Development Mode

```bash
# Activate virtual environment
source venv/bin/activate

# Run with auto-reload (for development)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Code Structure

- `main.py`: Contains the FastAPI application with endpoints and static file serving
- `index.html`: Frontend interface with JavaScript for API interaction
- Static files are served from the root directory

## 🧪 Testing

### Manual Testing

1. Start the application: `python main.py`
2. Open http://localhost:8000/ in your browser
3. Open browser developer console (F12)
4. Check for the API response log: `{"message": "Hello, Recipe App!"}`

### API Testing

Use the interactive documentation at http://localhost:8000/docs to test endpoints directly.

## 📝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes
4. Commit your changes: `git commit -m 'Add new feature'`
5. Push to the branch: `git push origin feature/new-feature`
6. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Support

If you encounter any issues or have questions:

1. Check the [API documentation](http://localhost:8000/docs)
2. Review the browser console for JavaScript errors
3. Check the server logs for backend errors

## 🔮 Future Enhancements

- [ ] Recipe database integration
- [ ] User authentication
- [ ] Recipe search and filtering
- [ ] Image upload for recipes
- [ ] Recipe sharing functionality
- [ ] Mobile-responsive design improvements 