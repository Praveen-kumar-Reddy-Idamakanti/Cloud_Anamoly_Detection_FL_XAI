
import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  AlertTriangle,
  Upload,
  LineChart,
  Shield,
  Settings,
  ChevronRight,
  BookOpen,
  Users,
  Database,
} from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';
import { cn } from '@/lib/utils';

interface SidebarProps {
  isSidebarOpen: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({ isSidebarOpen }) => {
  const { user } = useAuth();
  const isAdmin = user?.role === 'admin';

  const menuItems = [
    {
      title: 'Dashboard',
      path: '/dashboard',
      icon: LayoutDashboard,
    },
    {
      title: 'Anomalies',
      path: '/anomalies',
      icon: AlertTriangle,
    },
    {
      title: 'Upload Logs',
      path: '/upload',
      icon: Upload,
    },
    {
      title: 'Analytics',
      path: '/analytics',
      icon: LineChart,
    },
    {
      title: 'XAI Explanations',
      path: '/explanations',
      icon: BookOpen,
    },
  ];

  const adminItems = [
    {
      title: 'User Management',
      path: '/users',
      icon: Users,
    },
    {
      title: 'Model Management',
      path: '/models',
      icon: Database,
    },
  ];

  const renderNavItem = (item: { title: string; path: string; icon: any }) => {
    const Icon = item.icon;
    
    return (
      <NavLink
        key={item.path}
        to={item.path}
        className={({ isActive }) =>
          cn(
            'flex items-center gap-3 px-3 py-2 rounded-md transition-colors',
            isActive
              ? 'bg-cyberpurple text-white'
              : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
          )
        }
      >
        <Icon className="h-5 w-5" />
        <span className={cn('transition-opacity', isSidebarOpen ? 'opacity-100' : 'opacity-0 hidden')}>
          {item.title}
        </span>
      </NavLink>
    );
  };

  return (
    <aside
      className={cn(
        'fixed inset-y-0 left-0 z-30 flex flex-col bg-background border-r border-border transition-all duration-300 ease-in-out py-14',
        isSidebarOpen ? 'w-64' : 'w-16'
      )}
    >
      <div className="flex-1 overflow-y-auto py-5 px-3">
        <div className="mb-6 px-3">
          <div className="flex items-center justify-center mb-6">
            <Shield className={cn('h-8 w-8 text-cyberpurple', !isSidebarOpen && 'mx-auto')} />
            {isSidebarOpen && <span className="ml-3 font-semibold text-xl text-gradient">CloudShield</span>}
          </div>
        </div>

        <nav className="space-y-1">
          {menuItems.map(renderNavItem)}

          {isAdmin && (
            <>
              <div className={cn('mt-6 mb-2 px-3', !isSidebarOpen && 'text-center')}>
                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  {isSidebarOpen ? 'Admin' : ''}
                </div>
              </div>
              {adminItems.map(renderNavItem)}
            </>
          )}
        </nav>
      </div>

      <div className="p-3 border-t border-border">
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            cn(
              'flex items-center gap-3 px-3 py-2 rounded-md transition-colors',
              isActive
                ? 'bg-cyberpurple text-white'
                : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
            )
          }
        >
          <Settings className="h-5 w-5" />
          <span className={cn('transition-opacity', isSidebarOpen ? 'opacity-100' : 'opacity-0 hidden')}>
            Settings
          </span>
        </NavLink>
      </div>
    </aside>
  );
};

export default Sidebar;
