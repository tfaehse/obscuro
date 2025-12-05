export type ToastType = 'info' | 'success' | 'error';

export class Toast {
    private static container: HTMLElement | null = null;

    private static ensureContainer() {
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.id = 'toast-container';
            this.container.style.position = 'fixed';
            this.container.style.bottom = '20px';
            this.container.style.right = '20px';
            this.container.style.display = 'flex';
            this.container.style.flexDirection = 'column';
            this.container.style.gap = '10px';
            this.container.style.zIndex = '10000';
            document.body.appendChild(this.container);
        }
    }

    static show(message: string, type: ToastType = 'info', duration: number = 5000) {
        this.ensureContainer();

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;

        // Basic styles
        toast.style.padding = '12px 20px';
        toast.style.borderRadius = '6px';
        toast.style.color = '#fff';
        toast.style.fontSize = '14px';
        toast.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s ease-in-out';
        toast.style.minWidth = '250px';
        toast.style.maxWidth = '400px';

        // Type-specific colors (using CSS variables if available, falling back to hex)
        if (type === 'error') {
            toast.style.backgroundColor = 'var(--danger, #dc3545)';
        } else if (type === 'success') {
            toast.style.backgroundColor = 'var(--ok, #28a745)';
        } else {
            toast.style.backgroundColor = 'var(--primary, #007bff)';
        }

        this.container!.appendChild(toast);

        // Fade in
        requestAnimationFrame(() => {
            toast.style.opacity = '1';
        });

        // Auto remove
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => {
                if (toast.parentElement) {
                    toast.parentElement.removeChild(toast);
                }
            }, 300);
        }, duration);
    }
}
