
import React, { useState, useEffect } from 'react';
import { Users } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { useAuth } from '../contexts/AuthContext';
import Navbar from '../components/Layout/Navbar';
import Sidebar from '../components/Layout/Sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import UserTable from '../components/UserManagement/UserTable';
import AddUserDialog from '../components/UserManagement/AddUserDialog';
import UserSearchBar from '../components/UserManagement/UserSearchBar';
import { useUserManagement } from '../hooks/useUserManagement';

const UserManagement: React.FC = () => {
  const { isAuthenticated, user } = useAuth();
  const navigate = useNavigate();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const {
    users,
    searchQuery,
    setSearchQuery,
    isAddUserOpen,
    setIsAddUserOpen,
    newUser,
    setNewUser,
    handleAddUser,
    updateUserStatus,
    updateUserRole
  } = useUserManagement();

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login');
      return;
    }

    if (user?.role !== 'admin') {
      navigate('/dashboard');
      toast.error('You do not have access to this page');
      return;
    }
  }, [isAuthenticated, navigate, user]);

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  return (
    <div className="min-h-screen">
      <Navbar toggleSidebar={toggleSidebar} />
      <Sidebar isSidebarOpen={isSidebarOpen} />

      <main className={`pt-16 transition-all duration-300 ${isSidebarOpen ? 'md:ml-64' : 'md:ml-16'}`}>
        <div className="p-4 md:p-6 max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <Users className="h-6 w-6 mr-2 text-cyberpurple" />
              <h1 className="text-2xl font-bold">User Management</h1>
            </div>
            
            <div className="flex items-center gap-2">
              <UserSearchBar 
                searchQuery={searchQuery} 
                onSearchChange={setSearchQuery} 
              />
              
              <AddUserDialog
                isOpen={isAddUserOpen}
                onOpenChange={setIsAddUserOpen}
                newUser={newUser}
                onNewUserChange={setNewUser}
                onAddUser={handleAddUser}
              />
            </div>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Users</CardTitle>
              <CardDescription>
                Manage user accounts and access permissions.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <UserTable 
                users={users} 
                onUpdateUserStatus={updateUserStatus} 
                onUpdateUserRole={updateUserRole} 
              />
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default UserManagement;
