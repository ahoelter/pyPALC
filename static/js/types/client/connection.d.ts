import { Message } from "../protocol/message";
import { Receiver } from "../protocol/receiver";
import { ClientSession } from "./session";
export declare const DEFAULT_SERVER_WEBSOCKET_URL = "ws://localhost:5006/ws";
export declare const DEFAULT_SESSION_ID = "default";
export declare type Rejecter = (error: Error | string) => void;
export declare class ClientConnection {
    readonly url: string;
    readonly id: string;
    readonly args_string: string | null;
    protected _on_have_session_hook: ((session: ClientSession) => void) | null;
    protected _on_closed_permanently_hook: (() => void) | null;
    protected readonly _number: number;
    socket: WebSocket | null;
    session: ClientSession | null;
    closed_permanently: boolean;
    protected _current_handler: ((message: Message) => void) | null;
    protected _pending_ack: [(connection: ClientConnection) => void, Rejecter] | null;
    protected _pending_replies: {
        [key: string]: [(message: Message) => void, Rejecter];
    };
    protected readonly _receiver: Receiver;
    constructor(url?: string, id?: string, args_string?: string | null, _on_have_session_hook?: ((session: ClientSession) => void) | null, _on_closed_permanently_hook?: (() => void) | null);
    connect(): Promise<ClientConnection>;
    close(): void;
    protected _schedule_reconnect(milliseconds: number): void;
    send(message: Message): void;
    send_with_reply(message: Message): Promise<Message>;
    protected _pull_doc_json(): Promise<Message>;
    protected _repull_session_doc(): void;
    protected _on_open(resolve: (connection: ClientConnection) => void, reject: Rejecter): void;
    protected _on_message(event: MessageEvent): void;
    protected _on_close(event: CloseEvent): void;
    protected _on_error(reject: Rejecter): void;
    protected _close_bad_protocol(detail: string): void;
    protected _awaiting_ack_handler(message: Message): void;
    protected _steady_state_handler(message: Message): void;
}
export declare function pull_session(url?: string, session_id?: string, args_string?: string): Promise<ClientSession>;
