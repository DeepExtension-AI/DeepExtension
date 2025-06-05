CREATE TYPE public.quantize_enum AS ENUM (
    'q4_K_M',
    'q4_K_S',
    'q8_0',
    'None'
    );


ALTER TYPE public.quantize_enum OWNER TO postgres;

CREATE TYPE public.model_stage AS ENUM (
    'base',
    'trained',
    'saved',
    'deployed'
    );


ALTER TYPE public.model_stage OWNER TO postgres;
