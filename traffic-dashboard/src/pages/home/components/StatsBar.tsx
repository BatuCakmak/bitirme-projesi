import { useState, useEffect } from 'react';
import { useThemeColors } from '@/hooks/useThemeColors';

export default function StatsBar() {
  const t = useThemeColors();
  
  // API'den gelecek verileri tutacağımız state
  const [liveData, setLiveData] = useState({
    toplamArac: 0,
    akisHizi: 0,
    aktifKamera: 44, // Şimdilik statik
    aktifUyari: 3    // Şimdilik statik
  });

  // Animasyon için ekranda görünen sayılar
  const [counts, setCounts] = useState([0, 0, 44, 3]);

  // 1. API'den Canlı Verileri Çekme İşlemi
  useEffect(() => {
    const fetchStats = async () => {
      try {
        // İstatistikleri çek (Toplam araç vs.)
        const statsRes = await fetch('http://localhost:8080/api/vehicles/stats');
        if (statsRes.ok) {
          const statsJson = await statsRes.json();
          
          // Akış hızını çek
          const flowRes = await fetch('http://localhost:8080/api/vehicles/flow-rate');
          const flowRate = flowRes.ok ? parseInt(await flowRes.text()) : 0;

          // State'i güncelle
          setLiveData(prev => ({
            ...prev,
            toplamArac: statsJson.toplamArac || 0,
            akisHizi: flowRate
          }));
        }
      } catch (error) {
        console.error("API Bağlantı Hatası:", error);
      }
    };

    // İlk açılışta hemen çek
    fetchStats();

    // Sonrasında her 3 saniyede bir verileri güncelle
    const interval = setInterval(fetchStats, 3000);
    return () => clearInterval(interval);
  }, []);

  // 2. Sayı Yükselme Animasyonu (Eski mantığını canlı veriye uyarladık)
  useEffect(() => {
    const targetValues = [liveData.toplamArac, liveData.akisHizi, liveData.aktifKamera, liveData.aktifUyari];
    
    const timers = targetValues.map((targetValue, i) =>
      setInterval(() => {
        setCounts(prev => {
          const next = [...prev];
          // Eğer hedef değere ulaşmadıysa artır (veya aniden düştüyse eşitle)
          if (next[i] < targetValue) {
            next[i] = Math.min(next[i] + Math.max(1, Math.ceil((targetValue - next[i]) / 10)), targetValue);
          } else if (next[i] > targetValue) {
             next[i] = targetValue; // Veritabanı sıfırlanırsa direkt düşsün
          }
          return next;
        });
      }, 50)
    );
    return () => timers.forEach(clearInterval);
  }, [liveData]); // liveData değiştikçe animasyonu tetikle

  // UI için kart konfigürasyonunu dinamikleştiriyoruz
  const statsConfig = [
    { label: 'Toplam Araç', value: counts[0], unit: '', icon: 'ri-car-line', color: '#00d4ff', change: 'Canlı', up: true },
    { label: 'Akış Hızı (Araç/Dk)', value: counts[1], unit: '', icon: 'ri-bar-chart-fill', color: '#ffd700', change: 'Canlı', up: true },
    { label: 'Aktif Kameralar', value: counts[2], unit: '/48', icon: 'ri-camera-line', color: '#a855f7', change: 'Stabil', up: null },
    { label: 'Aktif Uyarı', value: counts[3], unit: '', icon: 'ri-alarm-warning-line', color: '#ff4757', change: '+1', up: true },
  ];

  return (
    <div className="grid grid-cols-4 gap-3">
      {statsConfig.map((stat) => (
        <div key={stat.label} className={`${t.cardBg} rounded-xl p-4 border ${t.border} transition-colors duration-300`}>
          <div className="flex items-start justify-between mb-3">
            <div className="w-10 h-10 flex items-center justify-center rounded-lg" style={{ backgroundColor: `${stat.color}20` }}>
              <i className={`${stat.icon} text-lg`} style={{ color: stat.color }}></i>
            </div>
            <div className={`flex items-center gap-1 text-xs font-medium ${
              stat.up === null ? t.textSecondary : stat.up ? 'text-[#00c97a]' : 'text-[#ff4757]'
            }`}>
              {stat.up !== null && <i className={stat.up ? 'ri-arrow-up-line text-xs' : 'ri-arrow-down-line text-xs'}></i>}
              {stat.change}
            </div>
          </div>
          <div className={`${t.textPrimary} font-bold text-2xl font-mono`}>
            {stat.value.toLocaleString('tr-TR')}{stat.unit}
          </div>
          <div className={`${t.textSecondary} text-xs mt-1`}>{stat.label}</div>
        </div>
      ))}
    </div>
  );
}