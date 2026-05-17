import type { RouteObject } from "react-router-dom";
import NotFound from "../pages/NotFound";
import Home from "../pages/home/page";
import VehicleCount from "../pages/vehicle-count/page";
import VehicleTypes from "../pages/vehicle-types/page";
import TrafficAnalysis from "../pages/traffic-analysis/page";
import AlternativeRoutes from "../pages/alternative-routes/page";
import Settings from "../pages/settings/page";

const routes: RouteObject[] = [
  { path: "/", element: <Home /> },
  { path: "/vehicle-count", element: <VehicleCount /> },
  { path: "/vehicle-types", element: <VehicleTypes /> },
  { path: "/traffic-analysis", element: <TrafficAnalysis /> },
  { path: "/alternative-routes", element: <AlternativeRoutes /> },
  { path: "/settings", element: <Settings /> },
  { path: "*", element: <NotFound /> },
];

export default routes;
