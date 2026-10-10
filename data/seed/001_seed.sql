CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_name TEXT NOT NULL,
    status TEXT NOT NULL,
    total_cents INTEGER NOT NULL,
    currency CHAR(3) NOT NULL DEFAULT 'INR',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS payments (
    payment_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    amount_cents INTEGER NOT NULL,
    currency CHAR(3) NOT NULL DEFAULT 'INR',
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

INSERT INTO orders (order_id, customer_name, status, total_cents, currency)
VALUES
    ('12345', 'Asha Patel', 'paid', 499900, 'INR'),
    ('12346', 'Rohan Mehta', 'shipped', 129900, 'INR')
ON CONFLICT (order_id) DO NOTHING;

INSERT INTO payments (payment_id, order_id, amount_cents, currency, status, created_at)
VALUES
    ('pay-12345-a', '12345', 499900, 'INR', 'succeeded', '2026-09-12T10:00:00Z'),
    ('pay-12345-b', '12345', 499900, 'INR', 'succeeded', '2026-09-12T10:02:00Z'),
    ('pay-12346-a', '12346', 129900, 'INR', 'succeeded', '2026-09-14T08:30:00Z')
ON CONFLICT (payment_id) DO NOTHING;
