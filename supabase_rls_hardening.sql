-- RLS hardening patch for existing Lego_Hunter Supabase projects.
-- Execute once in Supabase SQL Editor.

alter table if exists public.opportunity_radar enable row level security;
alter table if exists public.market_time_series enable row level security;
alter table if exists public.portfolio enable row level security;
alter table if exists public.fiscal_log enable row level security;

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
