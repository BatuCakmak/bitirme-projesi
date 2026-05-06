import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

// --- TASARIM VE RENK SABİTLERİ ---
const THEME = {
  primary: '#1d4ed8', 
  background: '#f8fafc', 
  cardBg: '#ffffff', 
  textMain: '#0f172a', 
  textSub: '#64748b', 
  status: {
    free: { bg: '#d1fae5', text: '#065f46', border: '#10b981', title: "AKICI" },
    heavy: { bg: '#fef3c7', text: '#92400e', border: '#f59e0b', title: "YOĞUN" },
    stuck: { bg: '#fee2e2', text: '#991b1b', border: '#ef4444', title: "SIKIŞIK (KİLİTLİ)" },
    empty: { bg: '#e2e8f0', text: '#1e293b', border: '#64748b', title: "YOL BOŞ" }
  },
  shadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)', 
  radius: '16px' 
};

const CHART_COLORS = {
  'Araba': '#3b82f6',
  'Motor': '#8b5cf6',
  'Kamyon': '#f97316',
  'Otobus': '#10b981',
  'Agir Vasita': '#f59e0b'
};

// --- STİLLER ---
const styles = {
  mainContainer: {
    fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
    backgroundColor: THEME.background,
    minHeight: '100vh',
    padding: '40px 20px'
  },
  header: { textAlign: 'center', marginBottom: '40px' },
  mainTitle: { fontSize: '2.5rem', fontWeight: '800', color: THEME.textMain, margin: 0, letterSpacing: '-1px' },
  subTitle: { color: THEME.textSub, fontSize: '1.1rem', marginTop: '10px' },
  
  statsRow: { display: 'flex', justifyContent: 'center', gap: '20px', marginBottom: '40px', flexWrap: 'wrap' },
  statCard: { 
    backgroundColor: THEME.cardBg, borderRadius: THEME.radius, boxShadow: THEME.shadow, 
    padding: '24px', width: '220px', textAlign: 'center' 
  },
  statusCard: (config) => ({
    backgroundColor: config.bg, borderRadius: THEME.radius, boxShadow: THEME.shadow,
    padding: '24px', width: '320px', textAlign: 'center', color: config.text,
    borderLeft: `10px solid ${config.border}`
  }),

  // Grafik ve Liste Bölümü
  chartSection: {
    display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: THEME.cardBg, borderRadius: THEME.radius, boxShadow: THEME.shadow,
    padding: '40px', maxWidth: '1100px', margin: '0 auto', gap: '40px', flexWrap: 'wrap'
  },
  chartContainer: { flex: 1, height: '400px', minWidth: '300px' },
  
  // Rakamsal Liste Stilleri
  dataList: { flex: 1, minWidth: '300px', display: 'flex', flexDirection: 'column', gap: '15px' },
  dataItem: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '15px 20px', borderRadius: '12px', backgroundColor: '#f1f5f9',
    transition: 'transform 0.2s'
  },
  itemLabel: { display: 'flex', alignItems: 'center', fontWeight: '600', color: THEME.textMain },
  colorDot: (color) => ({ height: '12px', width: '12px', borderRadius: '50%', backgroundColor: color, marginRight: '12px' }),
  itemValue: { fontWeight: '800', fontSize: '1.2rem', color: THEME.textMain },
  percentageLabel: { fontSize: '0.85rem', color: THEME.textSub, marginLeft: '8px', fontWeight: '400' }
};

function App() {
  const [stats, setStats] = useState([]);
  const [flowRate, setFlowRate] = useState(0);
  const [density, setDensity] = useState(0); // YENİ: Yoğunluk State'i

  useEffect(() => {
    const fetchData = () => { 
      fetchStats(); 
      fetchFlowRate(); 
      fetchDensity(); // YENİ: Yoğunluğu da çek
    };
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get('http://localhost:8080/api/vehicles/stats');
      setStats(response.data.map(item => ({ name: item[0], count: item[1] })));
    } catch (e) { console.error(e); }
  };

  const fetchFlowRate = async () => {
    try {
      const response = await axios.get('http://localhost:8080/api/vehicles/flow-rate');
      setFlowRate(response.data);
    } catch (e) { console.error(e); }
  };

  const fetchDensity = async () => {
    try {
      const response = await axios.get('http://localhost:8080/api/vehicles/current-density');
      setDensity(response.data);
    } catch (e) { console.error(e); }
  };

  const totalVehicles = stats.reduce((sum, item) => sum + item.count, 0);

  // --- GELİŞMİŞ MFD (Makroskopik Temel Diyagram) MANTIĞI ---
  let currentStatus = THEME.status.free; // Varsayılan: Akıcı

  if (density > 15 && flowRate < 5) {
    // Ekranda çok araç var ama geçen yok -> Trafik felç
    currentStatus = THEME.status.stuck;
  } else if (density > 10 && flowRate >= 5) {
    // Çok araç var, hareket de var -> Yoğun
    currentStatus = THEME.status.heavy;
  } else if (density <= 2 && flowRate <= 2) {
    // Ekranda da yok, geçen de yok -> Yol boş
    currentStatus = THEME.status.empty;
  }

  return (
    <div style={styles.mainContainer}>
      
      <header style={styles.header}>
        <h1 style={styles.mainTitle}>🚦 Akıllı Şehir Trafik Analiz Sistemi</h1>
        <p style={styles.subTitle}>Canlı Veri ve Araç Sınıflandırma Paneli (MFD Analizli)</p>
      </header>
      
      {/* KARTLARIN OLDUĞU ÜST BÖLÜM */}
      <div style={styles.statsRow}>
        
        {/* TOPLAM GEÇİŞ KARTI */}
        <div style={styles.statCard}>
          <h4 style={{margin: 0, color: THEME.textSub}}>TOPLAM GEÇİŞ</h4>
          <p style={{fontSize: '3rem', fontWeight: '900', margin: '10px 0', color: THEME.textMain}}>{totalVehicles}</p>
        </div>

        {/* AKIŞ (FLOW RATE) KARTI */}
        <div style={styles.statCard}>
          <h4 style={{margin: 0, color: THEME.textSub}}>ANLIK AKIŞ (q)</h4>
          <p style={{fontSize: '3rem', fontWeight: '900', margin: '10px 0', color: '#3b82f6'}}>{flowRate}</p>
          <span style={{fontSize: '0.9rem', color: THEME.textSub}}>Araç / Dakika</span>
        </div>

        {/* YENİ: YOĞUNLUK (DENSITY) KARTI */}
        <div style={styles.statCard}>
          <h4 style={{margin: 0, color: THEME.textSub}}>YOĞUNLUK (k)</h4>
          <p style={{fontSize: '3rem', fontWeight: '900', margin: '10px 0', color: '#8b5cf6'}}>{density}</p>
          <span style={{fontSize: '0.9rem', color: THEME.textSub}}>Ekranda Bekleyen</span>
        </div>

        {/* DURUM KARTI */}
        <div style={styles.statusCard(currentStatus)}>
          <h4 style={{margin: 0}}>YOL DURUMU</h4>
          <p style={{fontSize: '2rem', fontWeight: '900', margin: '10px 0'}}>{currentStatus.title}</p>
        </div>

      </div>
      {/* CANLI KAMERA YAYINI BÖLÜMÜ */}
      <div style={{
        maxWidth: '1100px', margin: '0 auto 40px auto', backgroundColor: THEME.cardBg, 
        borderRadius: THEME.radius, boxShadow: THEME.shadow, padding: '20px', textAlign: 'center'
      }}>
        <h3 style={{margin: '0 0 15px 0', color: THEME.textMain, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
          <span style={styles.colorDot('#ef4444')}></span> Canlı Mobese Akışı (AI Destekli)
        </h3>
        <div style={{ borderRadius: '12px', overflow: 'hidden', backgroundColor: '#000', display: 'flex', justifyContent: 'center' }}>
          {/* Flask sunucusundan (Python) gelen canlı yayını okuyan etiket */}
          <img 
            src="http://localhost:5000/video_feed" 
            alt="Kamera Bağlantısı Bekleniyor..." 
            style={{ width: '100%', maxHeight: '600px', objectFit: 'contain' }} 
          />
        </div>
      </div>

      {/* GRAFİK VE LİSTE BÖLÜMÜ */}
      <div style={styles.chartSection}>
        
        {/* SOL: Donut Grafik */}
        <div style={styles.chartContainer}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={stats} cx="50%" cy="50%" 
                innerRadius={80} outerRadius={130} 
                paddingAngle={8} dataKey="count" stroke="none"
              >
                {stats.map((entry, index) => (
                  <Cell key={index} fill={CHART_COLORS[entry.name] || '#cbd5e1'} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: THEME.shadow }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* SAĞ: Rakamsal Dağılım Listesi */}
        <div style={styles.dataList}>
          <h3 style={{margin: '0 0 10px 0', color: THEME.textMain}}>Araç Dağılım Detayları</h3>
          {stats.length > 0 ? stats.map((item, index) => {
            const percentage = ((item.count / totalVehicles) * 100).toFixed(1);
            return (
              <div key={index} style={styles.dataItem}>
                <div style={styles.itemLabel}>
                  <div style={styles.colorDot(CHART_COLORS[item.name])}></div>
                  {item.name}
                  <span style={styles.percentageLabel}>%{percentage}</span>
                </div>
                <div style={styles.itemValue}>
                  {item.count} <span style={{fontSize: '0.9rem', color: THEME.textSub, fontWeight: '400'}}>adet</span>
                </div>
              </div>
            );
          }) : <p style={{color: THEME.textSub}}>Henüz veri girişi yok...</p>}
        </div>

      </div>
    </div>
  );
}

export default App;