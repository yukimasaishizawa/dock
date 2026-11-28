import { initializeApp } from 'firebase/app';
import { browserLocalPersistence, getAuth, setPersistence } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { firebaseConfig } from './firebaseConfig';

export const firebaseApp = initializeApp(firebaseConfig);

export const auth = getAuth(firebaseApp);
export const db = getFirestore(firebaseApp);

// スマホのホーム画面から起動してもログインが続くように、端末に保存します。
// （失敗しても致命的ではないので握りつぶします）
void setPersistence(auth, browserLocalPersistence).catch(() => undefined);

/** Firestore のコレクション名。将来変えるならここだけ。 */
export const PLANS_COLLECTION = 'plans';
