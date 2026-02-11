-- Lego_Hunter Supabase Schema
-- Execute this script in Supabase SQL Editor.

create extension if not exists pgcrypto;

create or replace function public.touch_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create table if not exists public.opportunity_radar (
  id uuid primary key default gen_random_uuid(),
  set_id text not null,
  set_name text not null,
  theme text,
  source text not null,
  eol_date_prediction date,
  market_demand_score integer not null default 0 check (market_demand_score between 0 and 100),
  ai_investment_score integer not null default 1 check (ai_investment_score between 1 and 100),
  ai_analysis_summary text,
  current_price numeric(10, 2),
  currency text not null default 'EUR',
  discovered_at timestamptz not null default now(),
  last_seen_at timestamptz not null default now(),
  is_archived boolean not null default false,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (set_id, source)
);

create table if not exists public.market_time_series (
  id uuid primary key default gen_random_uuid(),
  recorded_at timestamptz not null default now(),
  set_id text not null,
  set_name text,
  platform text not null,
  listing_type text not null default 'unknown',
  price numeric(10, 2) not null check (price >= 0),
  shipping_cost numeric(10, 2) not null default 0,
  currency text not null default 'EUR',
  seller_name text,
  seller_rating numeric(4, 2),
  stock_status text,
  listing_url text,
  raw_payload jsonb not null default '{}'::jsonb
);

create table if not exists public.portfolio (
  id uuid primary key default gen_random_uuid(),
  set_id text not null unique,
  set_name text not null,
  theme text,
  purchase_date date not null default current_date,
  purchase_platform text,
  purchase_price numeric(10, 2) not null check (purchase_price >= 0),
  shipping_in_cost numeric(10, 2) not null default 0,
  quantity integer not null default 1 check (quantity > 0),
  estimated_market_price numeric(10, 2),
  status text not null default 'holding' check (status in ('holding', 'sold')),
  notes text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.fiscal_log (
  id uuid primary key default gen_random_uuid(),
  event_date date not null default current_date,
  platform text not null,
  event_type text not null check (event_type in ('buy', 'sell', 'fee', 'refund', 'tax')),
  set_id text,
  units integer not null default 1 check (units > 0),
  gross_amount numeric(10, 2) not null default 0,
  shipping_cost numeric(10, 2) not null default 0,
  fees numeric(10, 2) not null default 0,
  net_amount numeric(10, 2) generated always as (gross_amount - shipping_cost - fees) stored,
  notes text,
  created_at timestamptz not null default now()
);

create index if not exists idx_opportunity_radar_score
  on public.opportunity_radar (ai_investment_score desc, market_demand_score desc, last_seen_at desc)
  where is_archived = false;

create index if not exists idx_market_time_series_set_time
  on public.market_time_series (set_id, recorded_at desc);

create index if not exists idx_market_time_series_platform_time
  on public.market_time_series (platform, recorded_at desc);

create index if not exists idx_portfolio_status
  on public.portfolio (status, updated_at desc);

create index if not exists idx_fiscal_log_date_platform
  on public.fiscal_log (event_date, platform, event_type);

create trigger trg_opportunity_updated_at
before update on public.opportunity_radar
for each row execute function public.touch_updated_at();

create trigger trg_portfolio_updated_at
before update on public.portfolio
for each row execute function public.touch_updated_at();

-- Security hardening: enable RLS on all public tables exposed by PostgREST.
alter table public.opportunity_radar enable row level security;
alter table public.market_time_series enable row level security;
alter table public.portfolio enable row level security;
alter table public.fiscal_log enable row level security;

-- Service-role backend access only (GitHub Actions / webhook backend).
do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'opportunity_radar' and policyname = 'service_role_full_access_opportunity_radar'
  ) then
    create policy service_role_full_access_opportunity_radar
      on public.opportunity_radar
      for all
      to service_role
      using (true)
      with check (true);
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'market_time_series' and policyname = 'service_role_full_access_market_time_series'
  ) then
    create policy service_role_full_access_market_time_series
      on public.market_time_series
      for all
      to service_role
      using (true)
      with check (true);
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'portfolio' and policyname = 'service_role_full_access_portfolio'
  ) then
    create policy service_role_full_access_portfolio
      on public.portfolio
      for all
      to service_role
      using (true)
      with check (true);
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'fiscal_log' and policyname = 'service_role_full_access_fiscal_log'
  ) then
    create policy service_role_full_access_fiscal_log
      on public.fiscal_log
      for all
      to service_role
      using (true)
      with check (true);
  end if;
end $$;
