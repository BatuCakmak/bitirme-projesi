import { ReactNode } from 'react';
import Sidebar from './Sidebar';
import TopBar from './TopBar';
import { useThemeColors } from '@/hooks/useThemeColors';

interface DashboardLayoutProps {
  children: ReactNode;
  title: string;
  subtitle?: string;
}

export default function DashboardLayout({ children, title, subtitle }: DashboardLayoutProps) {
  const t = useThemeColors();

  return (
    <div className={`flex h-screen ${t.pageBg} overflow-hidden transition-colors duration-300`}>
      <Sidebar />
      <div className="flex-1 flex flex-col ml-[220px] overflow-hidden">
        <TopBar title={title} subtitle={subtitle} />
        <main className="flex-1 overflow-y-auto p-5">
          {children}
        </main>
      </div>
    </div>
  );
}
