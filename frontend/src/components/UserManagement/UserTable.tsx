
import React from 'react';
import { Shield, MoreHorizontal, Check, X } from 'lucide-react';
import { toast } from 'sonner';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table';
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  status: string;
}

interface UserTableProps {
  users: User[];
  onUpdateUserStatus: (id: string, status: string) => void;
  onUpdateUserRole: (id: string, role: string) => void;
}

const UserTable: React.FC<UserTableProps> = ({ 
  users, 
  onUpdateUserStatus, 
  onUpdateUserRole 
}) => {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Email</TableHead>
          <TableHead>Role</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {users.map((user) => (
          <TableRow key={user.id}>
            <TableCell className="font-medium">{user.name}</TableCell>
            <TableCell>{user.email}</TableCell>
            <TableCell>
              <Badge variant={
                user.role === 'admin' ? 'default' : 
                user.role === 'moderator' ? 'secondary' : 'outline'
              }>
                {user.role}
              </Badge>
            </TableCell>
            <TableCell>
              {user.status === 'active' ? (
                <Badge variant="secondary" className="bg-green-100 text-green-800 hover:bg-green-100">
                  Active
                </Badge>
              ) : (
                <Badge variant="destructive" className="bg-red-100 text-red-800 hover:bg-red-100">
                  Inactive
                </Badge>
              )}
            </TableCell>
            <TableCell>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon">
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuLabel>Actions</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => toast.info(`Edit ${user.name}`)}>
                    Edit user
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuLabel>Role</DropdownMenuLabel>
                  <DropdownMenuItem onClick={() => onUpdateUserRole(user.id, 'admin')}>
                    <Shield className="h-4 w-4 mr-2" />
                    Make Admin
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => onUpdateUserRole(user.id, 'moderator')}>
                    Make Moderator
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => onUpdateUserRole(user.id, 'user')}>
                    Make User
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  {user.status === 'active' ? (
                    <DropdownMenuItem onClick={() => onUpdateUserStatus(user.id, 'inactive')}>
                      <X className="h-4 w-4 mr-2 text-red-500" />
                      <span className="text-red-500">Deactivate</span>
                    </DropdownMenuItem>
                  ) : (
                    <DropdownMenuItem onClick={() => onUpdateUserStatus(user.id, 'active')}>
                      <Check className="h-4 w-4 mr-2 text-green-500" />
                      <span className="text-green-500">Activate</span>
                    </DropdownMenuItem>
                  )}
                </DropdownMenuContent>
              </DropdownMenu>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
};

export default UserTable;
