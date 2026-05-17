import { NavLink } from 'react-router-dom';
import { useThemeColors } from '@/hooks/useThemeColors';

const navItems = [
  { path: '/', icon: 'ri-map-2-line', label: 'Canlı Harita', exact: true },
  { path: '/vehicle-count', icon: 'ri-car-line', label: 'Araç Sayımı' },
  { path: '/vehicle-types', icon: 'ri-truck-line', label: 'Araç Türleri' },
  { path: '/traffic-analysis', icon: 'ri-bar-chart-2-line', label: 'Trafik Analizi' },
  { path: '/alternative-routes', icon: 'ri-route-line', label: 'Alternatif Rotalar' },
  { path: '/settings', icon: 'ri-settings-3-line', label: 'Ayarlar' },
];

export default function Sidebar() {
  const t = useThemeColors();

  return (
    <aside className={`fixed left-0 top-0 h-full w-[220px] ${t.sidebarBg} border-r ${t.border} flex flex-col z-40 transition-colors duration-300`}>
      {/* Logo */}
      <div className={`flex items-center gap-3 px-5 py-5 border-b ${t.border}`}>
        <img
          src="https://public.readdy.ai/ai/img_res/ddf3b20b-affc-4545-a351-4687bfd316a5.png"
          alt="UrbanEye Logo"
          className="w-9 h-9 object-contain"
        />
        <div>
          <div className={`${t.textPrimary} font-bold text-sm leading-tight`} style={{ fontFamily: "'Rajdhani', sans-serif" }}>UrbanEye</div>
          <div className={`${t.textAccent} text-[10px] font-medium tracking-widest uppercase`}>Trafik İzleme</div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 overflow-y-auto">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.exact}
            className={({ isActive }) =>
              `flex items-center gap-3 px-5 py-3 mx-2 rounded-lg mb-1 transition-all duration-200 cursor-pointer ${
                isActive
                  ? `${t.activeBg} ${t.navActive} border-l-2`
                  : `${t.navInactive} ${t.hoverBg} border-l-2 border-transparent`
              }`
            }
          >
            <div className="w-5 h-5 flex items-center justify-center">
              <i className={`${item.icon} text-base`}></i>
            </div>
            <span className="text-sm font-medium whitespace-nowrap">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* System Status */}
      <div className={`px-5 py-4 border-t ${t.border}`}>
        <div className={`${t.textSecondary} text-[10px] uppercase tracking-widest mb-3`}>Sistem Durumu</div>
        <div className="flex items-center gap-2 mb-2">
          <div className="w-2 h-2 rounded-full bg-[#00ff88] animate-pulse"></div>
          <span className={`${t.textSecondary} text-xs`}>44 Kamera Aktif</span>
        </div>
        <div className="flex items-center gap-2 mb-2">
          <div className="w-2 h-2 rounded-full bg-[#ffd700]"></div>
          <span className={`${t.textSecondary} text-xs`}>3 Uyarı</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#ff4757]"></div>
          <span className={`${t.textSecondary} text-xs`}>1 Çevrimdışı</span>
        </div>
      </div>
    </aside>
  );
}
