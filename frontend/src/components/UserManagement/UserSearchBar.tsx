
import React from 'react';
import { Search } from 'lucide-react';
import { Input } from '@/components/ui/input';

interface UserSearchBarProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
}

const UserSearchBar: React.FC<UserSearchBarProps> = ({ searchQuery, onSearchChange }) => {
  return (
    <div className="relative">
      <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
      <Input
        type="search"
        placeholder="Search users..."
        className="pl-8 w-[200px] md:w-[300px]"
        value={searchQuery}
        onChange={(e) => onSearchChange(e.target.value)}
      />
    </div>
  );
};

export default UserSearchBar;
