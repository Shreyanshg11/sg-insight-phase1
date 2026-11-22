import { createContext, useContext } from "react";

export const AuthContext = createContext({ enabled: false, signOut: async () => {} });
export const useOperatorAuth = () => useContext(AuthContext);
