import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { PlannerProvider, usePlanner } from './app/PlannerProvider';
import { ToastProvider } from './app/ToastProvider';
import { AppHeader } from './components/AppHeader';
import { Fab } from './components/Fab';
import { Splash } from './components/Splash';
import { DashboardScreen } from './screens/DashboardScreen';
import { DaySheet } from './screens/DaySheet';
import { LoginScreen } from './screens/LoginScreen';
import { NewPlanSheet } from './screens/NewPlanSheet';

function AppContent() {
  const { ready, session } = usePlanner();

  if (!ready) return <Splash />;

  // 認証が通るまでは、どの URL を開いてもログイン画面より先には進めません。
  if (!session) return <LoginScreen />;

  return (
    <div className="app-shell">
      <AppHeader />
      <DashboardScreen />
      <Fab />

      <Routes>
        <Route path="/" element={null} />
        <Route path="/day/:date" element={<DaySheet />} />
        <Route path="/new" element={<NewPlanSheet />} />
        <Route path="/login" element={<Navigate to="/" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <PlannerProvider>
        <ToastProvider>
          <AppContent />
        </ToastProvider>
      </PlannerProvider>
    </BrowserRouter>
  );
}
