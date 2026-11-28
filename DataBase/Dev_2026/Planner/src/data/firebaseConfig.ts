/**
 * Firebase の設定値。
 *
 * ここに入っている値はブラウザに配信される「公開情報」です（秘密鍵ではありません）。
 * 万一プロジェクトを切り替えるときは、`.env.local` に VITE_FIREBASE_* を書けば
 * そちらが優先されます。
 */
export interface FirebaseConfig {
  apiKey: string;
  authDomain: string;
  projectId: string;
  storageBucket: string;
  messagingSenderId: string;
  appId: string;
}

const env = import.meta.env;

export const firebaseConfig: FirebaseConfig = {
  apiKey: env.VITE_FIREBASE_API_KEY || 'AIzaSyCRFXXPRrulSDQ4FzkOsicNfHTc5BKXUtQ',
  authDomain: env.VITE_FIREBASE_AUTH_DOMAIN || 'planner-267b2.firebaseapp.com',
  projectId: env.VITE_FIREBASE_PROJECT_ID || 'planner-267b2',
  storageBucket: env.VITE_FIREBASE_STORAGE_BUCKET || 'planner-267b2.firebasestorage.app',
  messagingSenderId: env.VITE_FIREBASE_MESSAGING_SENDER_ID || '998139685902',
  appId: env.VITE_FIREBASE_APP_ID || '1:998139685902:web:0cca9a42568d925cb7605f',
};
