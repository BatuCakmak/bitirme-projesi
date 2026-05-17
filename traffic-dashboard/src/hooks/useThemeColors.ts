import { useTheme } from './useTheme';

export function useThemeColors() {
  const { isDark } = useTheme();

  return {
    // Backgrounds
    pageBg: isDark ? 'bg-[#0f1117]' : 'bg-[#f0f2f5]',
    cardBg: isDark ? 'bg-[#1a1d29]' : 'bg-white',
    sidebarBg: isDark ? 'bg-[#0d1117]' : 'bg-white',
    topbarBg: isDark ? 'bg-[#0d1117]' : 'bg-white',
    inputBg: isDark ? 'bg-[#0f1117]' : 'bg-[#f8f9fa]',
    hoverBg: isDark ? 'hover:bg-[#252836]' : 'hover:bg-[#f5f5f5]',
    activeBg: isDark ? 'bg-[#00d4ff]/10' : 'bg-[#0099bb]/10',

    // Borders
    border: isDark ? 'border-[#1e2433]' : 'border-[#e5e7eb]',
    divider: isDark ? 'divide-[#1e2433]' : 'divide-[#e5e7eb]',

    // Text
    textPrimary: isDark ? 'text-white' : 'text-[#1a1a2e]',
    textSecondary: isDark ? 'text-[#8892a4]' : 'text-[#6b7280]',
    textAccent: isDark ? 'text-[#00d4ff]' : 'text-[#0099bb]',

    // Nav active
    navActive: isDark ? 'text-[#00d4ff] border-[#00d4ff]' : 'text-[#0099bb] border-[#0099bb]',
    navInactive: isDark ? 'text-[#8892a4]' : 'text-[#6b7280]',

    // Accent color
    accent: isDark ? '#00d4ff' : '#0099bb',

    isDark,
  };
}
