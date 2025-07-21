
import { useState } from 'react';
import { toast } from 'sonner';

interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  status: string;
}

const mockUsers = [
  { id: '1', name: 'John Doe', email: 'john@example.com', role: 'admin', status: 'active' },
  { id: '2', name: 'Jane Smith', email: 'jane@example.com', role: 'user', status: 'active' },
  { id: '3', name: 'Bob Johnson', email: 'bob@example.com', role: 'user', status: 'inactive' },
  { id: '4', name: 'Alice Brown', email: 'alice@example.com', role: 'moderator', status: 'active' },
  { id: '5', name: 'Charlie Davis', email: 'charlie@example.com', role: 'user', status: 'active' },
];

export const useUserManagement = () => {
  const [users, setUsers] = useState<User[]>(mockUsers);
  const [searchQuery, setSearchQuery] = useState('');
  const [isAddUserOpen, setIsAddUserOpen] = useState(false);
  const [newUser, setNewUser] = useState({ name: '', email: '', role: 'user' });

  const filteredUsers = users.filter(user => 
    user.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
    user.email.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleAddUser = () => {
    if (!newUser.name || !newUser.email) {
      toast.error('Please fill all required fields');
      return;
    }

    const newId = (users.length + 1).toString();
    const userToAdd = {
      id: newId,
      name: newUser.name,
      email: newUser.email,
      role: newUser.role,
      status: 'active'
    };

    setUsers([...users, userToAdd]);
    setNewUser({ name: '', email: '', role: 'user' });
    setIsAddUserOpen(false);
    toast.success('User added successfully');
  };

  const updateUserStatus = (id: string, status: string) => {
    const updatedUsers = users.map(user => 
      user.id === id ? { ...user, status } : user
    );
    setUsers(updatedUsers);
    toast.success(`User ${status === 'active' ? 'activated' : 'deactivated'} successfully`);
  };

  const updateUserRole = (id: string, role: string) => {
    const updatedUsers = users.map(user => 
      user.id === id ? { ...user, role } : user
    );
    setUsers(updatedUsers);
    toast.success('User role updated successfully');
  };

  return {
    users: filteredUsers,
    searchQuery,
    setSearchQuery,
    isAddUserOpen,
    setIsAddUserOpen,
    newUser,
    setNewUser,
    handleAddUser,
    updateUserStatus,
    updateUserRole
  };
};
