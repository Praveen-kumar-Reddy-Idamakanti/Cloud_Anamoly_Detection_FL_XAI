# Cloud Anomaly Detection Frontend

A modern, responsive web application for monitoring and analyzing cloud infrastructure anomalies using Federated Learning and eXplainable AI (XAI).

![Dashboard Preview](public/dashboard-preview.png)

## Features

- **Real-time Monitoring**: Live anomaly detection and alerts
- **Interactive Dashboards**: Visualize system metrics and anomalies
- **XAI Explanations**: Understand model decisions with explainable AI
- **User Management**: Role-based access control
- **Model Management**: Deploy and manage ML models
- **Responsive Design**: Works on desktop and mobile devices

## Tech Stack

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS with Shadcn UI components
- **State Management**: React Query + Context API
- **Routing**: React Router v6
- **UI Components**: Radix UI Primitives + Shadcn UI
- **Charts**: Recharts
- **Forms**: React Hook Form with Zod validation
- **Authentication**: Supabase Auth
- **Real-time**: Supabase Realtime
- **Icons**: Lucide React
- **Notifications**: Sonner

## Prerequisites

- Node.js 18+ (LTS recommended)
- npm 9+ or pnpm 8+ or yarn 1.22+
- Git

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd frontend
```

### 2. Install Dependencies

Using npm:
```bash
npm install
```

Or using pnpm (recommended):
```bash
pnpm install
```

### 3. Environment Setup

Create a `.env` file in the root directory with the following variables:

```env
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_API_BASE_URL=your_api_base_url
```

### 4. Start Development Server

```bash
npm run dev
# or
pnpm dev
```

The application will be available at `http://localhost:5173`

## Project Structure

```
src/
├── api/               # API clients and services
├── assets/            # Static assets (images, fonts)
├── components/        # Reusable UI components
│   ├── ui/           # Shadcn UI components
│   ├── dashboard/    # Dashboard-specific components
│   └── layout/       # Layout components
├── contexts/         # React contexts
├── data/             # Mock data and types
├── hooks/            # Custom React hooks
├── integrations/     # Third-party integrations
├── lib/              # Utility functions
├── pages/            # Page components
└── styles/           # Global styles
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `VITE_SUPABASE_URL` | Supabase project URL | Yes |
| `VITE_SUPABASE_ANON_KEY` | Supabase anonymous key | Yes |
| `VITE_API_BASE_URL` | Base URL for API requests | Yes |
| `VITE_ENABLE_MOCK_API` | Enable mock API (true/false) | No |

## Authentication

The application uses Supabase Auth for authentication. To set up:

1. Create a new project in Supabase
2. Enable Email/Password authentication in the Auth providers
3. Configure the redirect URLs in Supabase Auth settings
4. Update the environment variables with your Supabase credentials

## API Integration

The frontend communicates with the following API endpoints:

- `/api/auth/*` - Authentication endpoints
- `/api/anomalies` - Anomaly detection data
- `/api/models` - Model management
- `/api/analytics` - Analytics data
- `/api/users` - User management

## Styling Guidelines

- Use Tailwind CSS utility classes for styling
- Follow the design system defined in `tailwind.config.ts`
- Use Shadcn UI components when possible
- Custom components should be placed in `src/components/ui`
- Use CSS modules for complex component-specific styles

## State Management

- **Server State**: Use React Query for data fetching and caching
- **Global UI State**: Use React Context for theme, auth state, etc.
- **Local State**: Use React's `useState` and `useReducer` for component-specific state

## Testing

Run tests with:

```bash
npm test
```

## Deployment

### Building for Production

```bash
npm run build
```

The build artifacts will be stored in the `dist/` directory.

### Deployment Options

1. **Vercel** (Recommended)
   - Connect your GitHub repository
   - Set up environment variables
   - Deploy with zero configuration

2. **Netlify**
   - Import your Git repository
   - Set build command: `npm run build`
   - Set publish directory: `dist`
   - Add environment variables

3. **Docker**
   ```dockerfile
   # Use the official Node.js image
   FROM node:18-alpine AS builder
   WORKDIR /app
   COPY package*.json ./
   RUN npm ci
   COPY . .
   RUN npm run build

   # Serve the built app using Nginx
   FROM nginx:alpine
   COPY --from=builder /app/dist /usr/share/nginx/html
   COPY nginx.conf /etc/nginx/conf.d/default.conf
   EXPOSE 80
   CMD ["nginx", "-g", "daemon off;"]
   ```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the repository or contact the development team.

---

Built with ❤️ by [Your Team Name]
