# Copilot Instructions

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Overview
This is a React application built with Vite for testing a ChatBot API. The application provides a user interface to:

- Send messages to a FastAPI chatbot backend
- View conversation history
- Manage multiple conversation threads
- Reset conversations

## Technical Stack
- **Frontend**: React 18 with Vite
- **Styling**: CSS modules or styled-components
- **HTTP Client**: Fetch API
- **Backend API**: FastAPI server running on http://localhost:8001

## API Endpoints
- `POST /chat?thread_id={id}` - Send a message to the chatbot
- `GET /chat/history/{thread_id}` - Get conversation history
- `POST /chat/reset/{thread_id}` - Reset a conversation thread

## Development Guidelines
- Use functional components with React hooks
- Implement proper error handling for API calls
- Follow React best practices for state management
- Ensure responsive design for mobile devices
- Add proper loading states for API interactions
