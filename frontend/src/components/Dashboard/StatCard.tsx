
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  description?: string;
  change?: number;
  isLoading?: boolean;
  variant?: 'default' | 'primary' | 'secondary' | 'danger' | 'warning' | 'success';
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  icon,
  description,
  change,
  isLoading = false,
  variant = 'default',
}) => {
  const getVariantClasses = () => {
    switch (variant) {
      case 'primary':
        return 'bg-cyberpurple/10 border-cyberpurple/20';
      case 'secondary':
        return 'bg-blue-500/10 border-blue-500/20';
      case 'danger':
        return 'bg-red-500/10 border-red-500/20';
      case 'warning':
        return 'bg-yellow-500/10 border-yellow-500/20';
      case 'success':
        return 'bg-green-500/10 border-green-500/20';
      default:
        return 'bg-secondary/40';
    }
  };

  const getIconClasses = () => {
    switch (variant) {
      case 'primary':
        return 'text-cyberpurple';
      case 'secondary':
        return 'text-blue-500';
      case 'danger':
        return 'text-red-500';
      case 'warning':
        return 'text-yellow-500';
      case 'success':
        return 'text-green-500';
      default:
        return 'text-muted-foreground';
    }
  };

  const getChangeColor = () => {
    if (!change) return 'text-muted-foreground';
    return change > 0 ? 'text-green-500' : 'text-red-500';
  };

  return (
    <Card className={cn('border overflow-hidden', getVariantClasses())}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <div className={cn("rounded-md p-2", getIconClasses())}>{icon}</div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-9 bg-muted rounded animate-pulse" />
        ) : (
          <div className="text-2xl font-bold">{value}</div>
        )}
        {description && (
          <p className="text-xs text-muted-foreground mt-1">{description}</p>
        )}
        {change !== undefined && (
          <div className={cn("text-xs font-medium mt-2", getChangeColor())}>
            {change > 0 ? '+' : ''}{change}% from last period
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default StatCard;
